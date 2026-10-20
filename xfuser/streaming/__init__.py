"""Continuous video streaming: chunk handoff, A/V stitching and an MPEG-TS server.

The generator (a torchrun job) writes finished chunks into a directory; the
streamer (``python -m xfuser.streaming.ts_server``) reads them in order, stitches
them into one continuous timeline and serves it over HTTP at real-time pace.
"""
