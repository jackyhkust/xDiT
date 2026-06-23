"""xFuser-wrapped Ideogram4Pipeline with asymmetric CFG parallelism.

When cfg_parallel_size == 2, rank 0 runs the unconditional transformer and
rank 1 runs the conditional transformer. Velocity predictions are gathered
and blended each step.
"""

import math
import torch

from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def _logit_normal_sigmas(num_inference_steps, mu, std=1.0, logsnr_min=-15.0, logsnr_max=18.0, device=None):
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    sigmas = (1.0 - t).flip(0)
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


def _resolution_aware_mu(height, width, base_mu, base_resolution=(512, 512)):
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


def _make_xfuser_ideogram4_pipeline_class():
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
        Ideogram4Pipeline,
        Ideogram4PipelineOutput,
        _expand_tensor_to_effective_batch,
    )
    from diffusers.utils.torch_utils import randn_tensor

    class xFuserIdeogram4Pipeline(Ideogram4Pipeline):

        @torch.no_grad()
        def __call__(self, *args, **kwargs):
            try:
                cfg_rank = get_classifier_free_guidance_rank()
                cfg_world_size = get_classifier_free_guidance_world_size()
            except (AssertionError, RuntimeError):
                cfg_rank = 0
                cfg_world_size = 1

            guidance_scale = kwargs.get("guidance_scale", None)
            guidance_schedule = kwargs.get("guidance_schedule", (7.0,) * 45 + (3.0,) * 3)
            has_guidance = guidance_scale is not None or (guidance_schedule is not None and any(g != 1.0 for g in guidance_schedule))
            do_cfg_parallel = has_guidance and cfg_world_size == 2

            if not do_cfg_parallel:
                return super().__call__(*args, **kwargs)

            return self._call_with_cfg_parallel(cfg_rank=cfg_rank, **kwargs)

        def _call_with_cfg_parallel(
            self,
            prompt=None,
            height=2048,
            width=2048,
            num_inference_steps=48,
            guidance_scale=None,
            guidance_schedule=(7.0,) * 45 + (3.0,) * 3,
            mu=0.0,
            std=1.5,
            max_sequence_length=2048,
            num_images_per_prompt=1,
            generator=None,
            latents=None,
            output_type="pil",
            return_dict=True,
            attention_kwargs=None,
            cfg_rank=0,
            **kwargs,
        ):
            self.check_inputs(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
            )

            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)

            device = self._execution_device
            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._interrupt = False

            grid_h = height // (self.vae_scale_factor * self.patch_size)
            grid_w = width // (self.vae_scale_factor * self.patch_size)
            num_image_tokens = grid_h * grid_w

            # Encode prompt (text encoder runs on all ranks, same result)
            llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
                prompt=prompt, grid_h=grid_h, grid_w=grid_w,
                max_sequence_length=max_sequence_length, device=device,
            )

            llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
            position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
            segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
            indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

            neg_llm_features = torch.zeros(
                batch_size * num_images_per_prompt, num_image_tokens,
                llm_features.shape[-1], dtype=llm_features.dtype, device=device,
            )
            neg_position_ids = position_ids[:, max_sequence_length:]
            neg_segment_ids = segment_ids[:, max_sequence_length:]
            neg_indicator = indicator[:, max_sequence_length:]

            schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
            sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
            self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            if guidance_scale is not None:
                guidance_schedule = [float(guidance_scale)] * num_inference_steps
            gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

            latent_dim = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_image_tokens=num_image_tokens,
                latent_dim=latent_dim,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
            )

            max_text_tokens = max_sequence_length
            text_z_padding = torch.zeros(
                batch_size * num_images_per_prompt, max_text_tokens, latent_dim,
                dtype=torch.float32, device=device,
            )

            llm_features = llm_features.to(self.transformer.dtype)
            neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)

            num_train_timesteps = self.scheduler.config.num_train_timesteps
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    t_model = 1.0 - (t.float() / num_train_timesteps)
                    t_model = t_model.expand(batch_size * num_images_per_prompt)

                    if cfg_rank == 1:
                        # Conditional pass
                        t_model_cond = t_model.to(self.transformer.dtype)
                        pos_z = torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype)
                        pos_out = self.transformer(
                            hidden_states=pos_z,
                            timestep=t_model_cond,
                            encoder_hidden_states=llm_features,
                            position_ids=position_ids,
                            segment_ids=segment_ids,
                            indicator=indicator,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        my_v = pos_out[:, max_text_tokens:].to(torch.float32)
                    else:
                        # Unconditional pass
                        t_model_uncond = t_model.to(self.unconditional_transformer.dtype)
                        neg_v = self.unconditional_transformer(
                            hidden_states=latents.to(self.unconditional_transformer.dtype),
                            timestep=t_model_uncond,
                            encoder_hidden_states=neg_llm_features,
                            position_ids=neg_position_ids,
                            segment_ids=neg_segment_ids,
                            indicator=neg_indicator,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0].to(torch.float32)
                        my_v = neg_v

                    # Gather velocities: rank 0 = neg_v, rank 1 = pos_v
                    uncond_v, cond_v = get_cfg_group().all_gather(my_v, separate_tensors=True)

                    self._guidance_scale = guidance_schedule[i]
                    gw_i = gw[i]
                    v = gw_i * cond_v + (1.0 - gw_i) * uncond_v

                    latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]
                    progress_bar.update()

            # Decode
            if output_type == "latent":
                image = latents
            else:
                z = latents
                bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
                bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
                bn_std = bn_std.to(device=z.device, dtype=z.dtype)
                z = z * bn_std + bn_mean

                patch = self.patch_size
                ae_channels = z.shape[-1] // (patch * patch)
                z = z.view(batch_size * num_images_per_prompt, grid_h, grid_w, patch, patch, ae_channels)
                z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
                z = z.view(batch_size * num_images_per_prompt, ae_channels, grid_h * patch, grid_w * patch)

                decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
                image = self.image_processor.postprocess(decoded.float(), output_type=output_type)

            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)
            return Ideogram4PipelineOutput(images=image)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            pipeline.__class__ = cls
            return pipeline

    return xFuserIdeogram4Pipeline


_pipeline_cls = None

def get_ideogram4_pipeline_class():
    global _pipeline_cls
    if _pipeline_cls is None:
        _pipeline_cls = _make_xfuser_ideogram4_pipeline_class()
    return _pipeline_cls
