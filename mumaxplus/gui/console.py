"""The console and "Copy as Python".

The console runs Python typed in the page, with the commands Run, Relax, ...; "Copy as
Python" gives what was done in the page as lines to add to the script.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import inspect
import io
import keyword
import re
import traceback

MAX_SUGGESTIONS = 1000  # enough for all attributes of a magnet, or of numpy
_NOT_POSITIONAL = re.compile(r"\s*(\*|[A-Za-z_]\w*\s*=(?!=))")  # a=1, *a, **a


def _blank_strings(text):
    """Return a line of Python, cut off at the cursor, with its strings blanked.

    Also returns what the line ends in: ``"code"``, ``"string"`` or ``"comment"``, where
    a comment is left out of the returned line.
    """
    code = list(text)
    quote = None
    skip = False
    for i, char in enumerate(text):
        if quote:
            code[i] = " "
            if skip:
                skip = False
            elif char == "\\":
                skip = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            code[i] = " "
        elif char == "#":
            return "".join(code[:i]), "comment"
    return "".join(code), "string" if quote else "code"


def _resolve(parts, ns):
    """Return the object with this dotted name, without running anything but getters."""
    obj = ns[parts[0]] if parts[0] in ns else getattr(builtins, parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _describe(name, value):
    """Return a suggestion for ``name``, whose ``value`` must not be computed for it."""
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if inspect.ismodule(value):
        kind = "module"
    elif inspect.isclass(value):
        kind = "class"
    elif callable(value):
        kind = "function"
    elif inspect.isdatadescriptor(value):
        kind = "property"
    else:  # a plain value: the docstring would be the one of its type
        return {"name": name, "kind": type(value).__name__, "doc": ""}
    doc = inspect.getdoc(value) or ""
    return {"name": name, "kind": kind, "doc": doc.partition("\n")[0]}


def complete(text, ns):
    """Return suggestions for the name that ``text``, a line up to the cursor, ends in.

    Returns a dict with the ``matches`` and the position ``start`` in ``text`` where the
    part they replace begins. This runs while typing, so no code in the line is run:
    only plain dotted names are looked up, like ``magnet.stray`` but not ``f().stray``.
    Going through an attribute, as in ``magnet.stray_field.``, does run its getter.
    """
    word = re.search(r"[\w.]*$", text)[0]
    start = len(text) - len(word.rpartition(".")[2])
    nothing = {"matches": [], "start": start}
    parts = word.split(".")
    prefix = parts.pop()
    if not word or not all(map(str.isidentifier, parts)):  # also .x (as in f().x), 1.5
        return nothing
    if prefix and not prefix.isidentifier():
        return nothing
    if _blank_strings(text)[1] != "code":
        return nothing

    if parts:
        try:
            owner = _resolve(parts, ns)
            names = dir(owner)
        except Exception:  # not defined (yet), or its getter failed
            return nothing
        lookup = lambda name: inspect.getattr_static(owner, name, None)
    else:
        if not prefix:  # nothing typed yet
            return nothing
        names = {*ns, *vars(builtins), *keyword.kwlist}
        lookup = lambda name: ns[name] if name in ns else vars(builtins).get(name)

    # exact case first, then other cases (run: Run); each with the user's names first
    low = prefix.lower()
    found = [n for n in names if n.lower().startswith(low)]
    if not prefix.startswith("_"):  # private names only when asked for
        found = [n for n in found if not n.startswith("_")]
    found.sort(
        key=lambda n: (not n.startswith(prefix), not parts and n not in ns, n.lower())
    )

    matches = []
    for name in found[:MAX_SUGGESTIONS]:
        if not parts and keyword.iskeyword(name):
            matches.append({"name": name, "kind": "keyword", "doc": ""})
            continue
        try:
            matches.append(_describe(name, lookup(name)))
        except Exception:
            matches.append({"name": name, "kind": "", "doc": ""})
    return {"matches": matches, "start": start}


def _open_call(code):
    """Return the name and state of the innermost call that ``code`` is still inside.

    Returns None if there is none, or if what is called is not a plain name. Brackets
    that only group or hold values are skipped: in ``f((1, [2,`` the call is ``f``.
    """
    stack = []
    for i, char in enumerate(code):
        if char in "([{":
            stack.append(
                {"bracket": char, "at": i, "arg": i + 1, "commas": 0, "named": False}
            )
        elif char in ")]}":
            if stack:
                stack.pop()
        elif char == "," and stack:
            top = stack[-1]
            # after f(a=1, or f(*a, the position no longer says which parameter it is
            if _NOT_POSITIONAL.match(code[top["arg"] : i]):
                top["named"] = True
            top["commas"] += 1
            top["arg"] = i + 1
    for top in reversed(stack):
        if top["bracket"] != "(":
            continue
        before = code[: top["at"]]
        name = re.search(r"[\w.]*$", before)[0]
        if name and not keyword.iskeyword(name):
            return name, top
        if before.rstrip().endswith((")", "]")):  # f()( or a[0](: needs running
            return None
    return None


def _signature_items(signature):
    """Return the parameters as text, and for each name its position in that list."""
    items, where = [], {}
    params = list(signature.parameters.values())
    has_star = any(p.kind == p.VAR_POSITIONAL for p in params)
    for i, param in enumerate(params):
        if param.kind == param.KEYWORD_ONLY and not has_star:
            items.append("*")
            has_star = True
        where[param.name] = len(items)
        items.append(str(param))
        following = params[i + 1].kind if i + 1 < len(params) else None
        if param.kind == param.POSITIONAL_ONLY and following != param.POSITIONAL_ONLY:
            items.append("/")
    return items, where


def _active_parameter(signature, code, call):
    """Return the name of the parameter the cursor is at, or None if that is unclear."""
    params = list(signature.parameters.values())
    current = code[call["arg"] :]
    named = re.match(r"\s*([A-Za-z_]\w*)\s*=(?!=)", current)
    if named:
        if named[1] in signature.parameters:
            return named[1]
        return next((p.name for p in params if p.kind == p.VAR_KEYWORD), None)
    if call["named"] or current.lstrip().startswith("*"):
        return None
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if call["commas"] < len(positional):
        return positional[call["commas"]].name
    return next((p.name for p in params if p.kind == p.VAR_POSITIONAL), None)


def call_hint(text, ns):
    """Return what to show about the call that ``text``, up to the cursor, is inside.

    Returns None if there is nothing to show, or else a dict with the ``name`` as typed,
    its ``params`` as text (None if unknown, as for compiled functions), the index
    ``active`` in them of the parameter at the cursor (or -1) and the first line of the
    ``doc``. As in :func:`complete`, nothing is called but the getters on the way.
    """
    code, ends_in = _blank_strings(text)
    call = _open_call(code)
    if ends_in == "comment" or call is None:
        return None
    name, state = call
    parts = name.split(".")
    if not all(map(str.isidentifier, parts)):  # also .x (as in f().x(), and 1(
        return None
    try:
        func = _resolve(parts, ns)
        if not callable(func):
            return None
        doc = (inspect.getdoc(func) or "").partition("\n")[0]
    except Exception:  # not defined (yet), or its getter failed
        return None
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):  # e.g. compiled: the doc usually starts with it
        return {"name": name, "params": None, "active": -1, "doc": doc} if doc else None
    items, where = _signature_items(signature)
    active = where.get(_active_parameter(signature, code, state), -1)
    return {"name": name, "params": items, "active": active, "doc": doc}


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

        def on_sim_thread(fn):  # the script can call them while the loop runs elsewhere
            @functools.wraps(fn)
            def call(*args, **kwargs):
                return self.call(lambda: fn(*args, **kwargs))

            return call

        helpers = [Run, Steps, Break, Relax, Minimize]
        helpers += [TableAdd, TableAutosave, TableSave, SetSolver]
        helpers = [on_sim_thread(fn) for fn in helpers]
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

    def _cmd_complete(self, c):
        text = c["text"]
        return {"ok": True, **complete(text, self.ns), "hint": call_hint(text, self.ns)}

    def _cmd_script(self, c):
        return {"ok": True, "script": self.script_text()}

    def _cmd_script_clear(self, c):
        self._record_run()  # forget the run so far too
        self.script, self._script_imports = [], []
        self._paint_undo = self._stroke = None  # they point at lines of the script
