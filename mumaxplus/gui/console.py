"""The console and "Copy as Python".

The console runs Python typed in the page, with the commands Run, Relax, ...; "Copy as
Python" gives what was done in the page as lines to add to the script.
"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback


class ConsoleMixin:
    """The Controller's console and recorded script."""

    def _init_script(self):
        self.script = []  # the recorded lines
        self._script_imports = []
        self._rec_t = self._rec_n = None  # time / step of the unrecorded part of a run

    def record(self, code, imports=()):
        """Add ``code`` to the recorded script, after the run done so far."""
        self._record_run()
        self.script.append(code)
        self._script_imports += [i for i in imports if i not in self._script_imports]

    def _did(self, code, imports=()):
        """Echo ``code`` in the console and record it."""
        self.say(code, "in")
        self.record(code, imports)

    def _record_run(self):
        """Record the part of the current run that was not recorded yet."""
        if not self.running or self._rec_t is None:
            return
        now = self._now()
        if self.steps_left is not None:
            n = self.steps_done - self._rec_n
            if n > 0:
                self.script.append(f"world.timesolver.steps({n})")
        else:
            # a finished Run(t) records t, not t plus the overshoot of the last step
            end = self.run_until
            if end is None or now < end * (1 - 1e-12):
                end = now
            if end > self._rec_t:
                self.script.append(f"world.timesolver.run({end - self._rec_t:.10g})")
        self._rec_t, self._rec_n = now, self.steps_done

    def script_text(self):
        """Return the recorded script, including the run so far."""
        self._record_run()
        head = [
            f"# What was done in the mumax+ GUI of {self.title}, as Python. Add these",
            "# lines to the end of the script, in place of gui.run(), to repeat it.",
        ]
        if not self.script:
            return "\n".join(head + ["# (nothing recorded yet)"]) + "\n"
        imports = self._script_imports and ["", *self._script_imports]
        return "\n".join(head + imports + [""] + self.script) + "\n"

    def _install_helpers(self):
        """Add the simulation commands to the console and the user's script."""

        def Run(t=None):
            self._start_run(t=None if t is None or t == float("inf") else float(t))

        def Steps(n):
            self._start_run(n=int(n))

        def Break():
            self._break()

        def Relax():
            self._blocking("relax", self.world.relax)

        def Minimize():
            self._blocking("minimize", self.world.minimize)

        def TableAdd(quantity, name=None, unit=None):
            self.table_add(quantity, name, unit)

        def TableAutosave(period):
            self._table_autosave(period)

        def TableSave():
            self.table.sample(self._now(), to_file=True)

        def SetSolver(method):
            self._set_method(method)
            self.record(f"world.timesolver.set_method({method!r})")

        helpers = [Run, Steps, Break, Relax, Minimize]
        helpers += [TableAdd, TableAutosave, TableSave, SetSolver]
        for fn in helpers:
            self.ns.setdefault(fn.__name__, fn)
        # the console lines that use them are not recorded as typed (Run, Relax, ...
        # record what they do themselves)
        self._helpers = {f.__name__ for f in helpers if self.ns[f.__name__] is f}

    def _cli(self, code):
        out = io.StringIO()
        self.say(code, "in")
        failed = False
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                try:
                    mode, tree = "eval", ast.parse(code, "<gui>", "eval")
                except SyntaxError:
                    mode, tree = "exec", ast.parse(code, "<gui>", "exec")
                self._record_run()
                at = len(self.script)  # before what Run, Relax, ... in it record
                result = eval(compile(tree, "<gui>", mode), self.ns)
                if result is not None:
                    print(repr(result))
                self._record_cli(code, tree, result, at)
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                lines = [f.lineno for f in tb if f.filename == "<gui>"]
                where = f" (line {lines[-1]})" if lines else ""
                print(f"{type(e).__name__}: {e}{where}")
                failed = True
        text = out.getvalue().rstrip()
        for line in text.splitlines():
            self.say(line, "err" if failed else "out")
        self.refresh_targets()
        self.describe()
        self._dirty = True
        return {"ok": not failed, "output": text}

    def _record_cli(self, code, tree, result, at):
        """Record a console line that worked, unless it only asked for a value."""
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if names & self._helpers:  # Run, Relax, ... record what they do themselves
            call = tree.body if isinstance(tree, ast.Expression) else None
            lone = (  # just one call of a GUI command, like Run(1e-9)
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in self._helpers
            )
            if not lone:
                self.script.insert(at, f"# console, uses GUI-only commands: {code}")
        elif isinstance(tree, ast.Module) or result is None:
            self.record(code)

    # ------------------------------------------------------------------- commands
    def _cmd_cli(self, c):
        return self._cli(c["code"])

    def _cmd_script(self, c):
        return {"ok": True, "script": self.script_text()}

    def _cmd_script_clear(self, c):
        self._record_run()  # forget the run so far too
        self.script, self._script_imports = [], []
