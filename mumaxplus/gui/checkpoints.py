"""Checkpoints: copies of the state (m, elastic fields, time) to go back to."""

from __future__ import annotations

import numpy as np


class CheckpointMixin:
    """The Controller's checkpoints, kept in host memory."""

    def _init_checkpoints(self):
        self.checkpoints = []
        self._next_checkpoint = 1

    def _state_vars(self):
        """(target, attribute) of every field that evolves in time."""
        out = []
        for name, obj in self.targets.items():
            if isinstance(getattr(type(obj), "magnetization", None), property):
                out.append((name, "magnetization"))
            if ":sub" not in name and getattr(obj, "enable_elastodynamics", False):
                out += [(name, "elastic_displacement"), (name, "elastic_velocity")]
        return out

    def _checkpoint(self):
        """Keep a copy of the state in memory, to go back to it with _restore."""
        ts = self._ts()
        state = [
            (n, a, np.array(getattr(self.targets[n], a).eval()))
            for n, a in self._state_vars()
        ]
        cp = {
            "id": self._next_checkpoint,
            "time": float(ts.time),
            "steps": self.steps_done,
            "mb": sum(arr.nbytes for _, _, arr in state) / 1e6,
            "state": state,
        }
        self._next_checkpoint += 1
        self.checkpoints.append(cp)
        refs = ", ".join(f"{self._ref(n)}.{a}.eval()" for n, a, _ in state)
        self._did(f"checkpoint_{cp['id']} = (world.timesolver.time, {refs})")
        self.say(f"checkpoint {cp['id']} at t = {cp['time']:.6g} s ({cp['mb']:.3g} MB)")

    def _restore(self, cid):
        """Go back to a checkpoint: its fields, time and step count."""
        cp = next((cp for cp in self.checkpoints if cp["id"] == cid), None)
        if cp is None:
            raise ValueError(f"no checkpoint {cid!r}")
        missing = [n for n, _, _ in cp["state"] if n not in self.targets]
        if missing:
            raise ValueError(f"checkpoint {cid}: magnet(s) gone: {', '.join(missing)}")
        self._break()
        ts = self._ts()
        for n, a, arr in cp["state"]:
            setattr(self.targets[n], a, arr)
        ts.time = cp["time"]
        self.steps_done = cp["steps"]
        self._fresh = True
        refs = ", ".join(f"{self._ref(n)}.{a}" for n, a, _ in cp["state"])
        self._did(f"world.timesolver.time, {refs} = checkpoint_{cid}")
        self.table.rewind(cp["time"])
        self._live_sample()
        for cfg in self.views:  # AutoSave: save now and every period from here
            cfg["_next_save"] = None
            if cfg.get("_movie"):  # a movie goes on from here too
                cfg["_movie"]["next"] = None
        self._autosave_tick()
        self._movie_tick()
        self._update_torque()
        self._dirty = True

    # ------------------------------------------------------------------- commands
    def _cmd_checkpoint(self, c):
        self._checkpoint()

    def _cmd_checkpoint_restore(self, c):
        self._restore(c["id"])

    def _cmd_checkpoint_remove(self, c):
        self.checkpoints = [cp for cp in self.checkpoints if cp["id"] != c["id"]]
