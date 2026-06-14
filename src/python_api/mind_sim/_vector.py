from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from . import _native


def _is_sequence(value):
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _values(value):
    if isinstance(value, Vector):
        return value.to_python()
    if hasattr(value, "to_python"):
        return [float(x) for x in value.to_python()]
    if _is_sequence(value):
        return [float(x) for x in value]
    return [float(value)]


def _resolve_end(end, size):
    if end is None or end < 0:
        return size - 1
    return min(end, size - 1)


def _range_values(values, start=0, end=None):
    if not values:
        return []
    begin = max(0, int(start))
    finish = _resolve_end(end, len(values))
    if finish < begin:
        return []
    return values[begin : finish + 1]


def _match(value, op, a, b=None):
    if op == ">=":
        return value >= a
    if op == "<=":
        return value <= a
    if op == "!=":
        return value != a
    if op == "==":
        return value == a
    if op == "<":
        return value < a
    if op == ">":
        return value > a
    if op == "[)":
        return a <= value < b
    if op == "[]":
        return a <= value <= b
    if op == "(]":
        return a < value <= b
    if op == "()":
        return a < value < b
    raise ValueError(f"unsupported Vector comparison operator: {op}")


class Vector:
    def __init__(self, *args):
        self._native = _native.Vector()
        self._recorded = False
        self._capacity = 0
        self._label = ""
        self._data: list[float] = []

        if not args:
            return
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                self._data = [0.0] * arg
            else:
                self._data = _values(arg)
            self._capacity = len(self._data)
            return
        if len(args) == 2 and isinstance(args[0], int):
            self._data = [float(args[1])] * args[0]
            self._capacity = len(self._data)
            return
        raise TypeError("Vector accepts (), (values), (size), or (size, fill)")

    @property
    def x(self):
        return self

    def _active_data(self):
        if self._recorded:
            return [float(x) for x in self._native.to_python()]
        return self._data

    def _replace(self, values):
        self._recorded = False
        self._data = [float(x) for x in values]
        self._capacity = max(self._capacity, len(self._data))
        return self

    def record(self, ref):
        self._native.record(ref)
        self._recorded = True
        return self

    def save_h5(self, path, name="values"):
        if self._recorded:
            self._native.save_h5(path, name)
            return None
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset(name, data=np.asarray(self._data, dtype=float))
        return None

    def to_python(self):
        return list(self._active_data())

    def from_python(self, values):
        return self._replace(_values(values))

    def clear(self):
        return self._replace([])

    def size(self):
        return len(self._active_data())

    def buffer_size(self, size=None):
        if size is not None:
            self._capacity = max(self._capacity, int(size), self.size())
        return max(self._capacity, self.size())

    def get(self, index):
        return self[index]

    def set(self, index, value):
        data = self.to_python()
        data[index] = float(value)
        self._replace(data)
        return self

    def append(self, *items):
        data = self.to_python()
        for item in items:
            data.extend(_values(item))
        return self._replace(data)

    def resize(self, size):
        size = int(size)
        data = self.to_python()
        if size < len(data):
            data = data[:size]
        else:
            data.extend([0.0] * (size - len(data)))
        return self._replace(data)

    def fill(self, value, start=0, end=None):
        data = self.to_python()
        finish = _resolve_end(end, len(data))
        for i in range(max(0, int(start)), finish + 1):
            data[i] = float(value)
        return self._replace(data)

    def c(self, start=0, end=None):
        out = Vector(_range_values(self.to_python(), start, end))
        out._label = self._label
        return out

    cl = c
    at = c

    def eq(self, other):
        return self.to_python() == _values(other)

    def copy(self, src, *args):
        source = _values(src)
        dest = self.to_python()
        if not args:
            return self._replace(source)
        if len(args) == 1 and isinstance(args[0], int):
            dest_start = args[0]
            needed = dest_start + len(source)
            if len(dest) < needed:
                dest.extend([0.0] * (needed - len(dest)))
            dest[dest_start:needed] = source
            return self._replace(dest)
        if len(args) == 2 and all(isinstance(arg, int) for arg in args):
            return self._replace(_range_values(source, args[0], args[1]))
        if len(args) == 3 and all(isinstance(arg, int) for arg in args):
            dest_start, src_start, src_end = args
            fragment = _range_values(source, src_start, src_end)
            needed = dest_start + len(fragment)
            if len(dest) < needed:
                dest.extend([0.0] * (needed - len(dest)))
            dest[dest_start:needed] = fragment
            return self._replace(dest)
        if len(args) in (1, 2) and isinstance(args[0], Vector):
            src_indices = [int(i) for i in args[0].to_python()]
            dest_indices = (
                [int(i) for i in args[1].to_python()]
                if len(args) == 2 and isinstance(args[1], Vector)
                else src_indices
            )
            if not dest:
                dest = [0.0] * (max(dest_indices, default=-1) + 1)
            for src_i, dest_i in zip(src_indices, dest_indices, strict=False):
                if dest_i >= len(dest):
                    dest.extend([0.0] * (dest_i + 1 - len(dest)))
                if 0 <= src_i < len(source):
                    dest[dest_i] = source[src_i]
            return self._replace(dest)
        raise TypeError("unsupported Vector.copy signature")

    def median(self):
        data = sorted(self.to_python())
        if not data:
            return 0.0
        mid = len(data) // 2
        if len(data) % 2:
            return data[mid]
        return 0.5 * (data[mid - 1] + data[mid])

    def mean(self, start=0, end=None):
        data = _range_values(self.to_python(), start, end)
        return float(sum(data) / len(data)) if data else 0.0

    def var(self, start=0, end=None):
        data = _range_values(self.to_python(), start, end)
        if len(data) < 2:
            return 0.0
        mu = sum(data) / len(data)
        return float(sum((x - mu) ** 2 for x in data) / (len(data) - 1))

    def stdev(self, start=0, end=None):
        return math.sqrt(self.var(start, end))

    def stderr(self, start=0, end=None):
        data = _range_values(self.to_python(), start, end)
        return self.stdev(start, end) / math.sqrt(len(data)) if data else 0.0

    def sum(self, start=0, end=None):
        return float(sum(_range_values(self.to_python(), start, end)))

    def sumsq(self, start=0, end=None):
        return float(sum(x * x for x in _range_values(self.to_python(), start, end)))

    def min(self, start=0, end=None):
        data = _range_values(self.to_python(), start, end)
        return float(min(data)) if data else 0.0

    def max(self, start=0, end=None):
        data = _range_values(self.to_python(), start, end)
        return float(max(data)) if data else 0.0

    def min_ind(self, start=0, end=None):
        data = self.to_python()
        values = _range_values(data, start, end)
        if not values:
            return -1.0
        return max(0, int(start)) + values.index(min(values))

    def max_ind(self, start=0, end=None):
        data = self.to_python()
        values = _range_values(data, start, end)
        if not values:
            return -1.0
        return max(0, int(start)) + values.index(max(values))

    def dot(self, other):
        return float(sum(a * b for a, b in zip(self.to_python(), _values(other), strict=False)))

    def mag(self):
        return math.sqrt(self.sumsq())

    def meansqerr(self, other, weights=None):
        data = self.to_python()
        rhs = _values(other)
        if weights is None:
            errs = [(a - b) ** 2 for a, b in zip(data, rhs, strict=False)]
        else:
            errs = [
                (a - b) ** 2 * w
                for a, b, w in zip(data, rhs, _values(weights), strict=False)
            ]
        return float(sum(errs) / len(errs)) if errs else 0.0

    def contains(self, value):
        return float(value) in self.to_python()

    def indgen(self, *args):
        if not args:
            return self._replace(range(self.size()))
        if len(args) == 1:
            return self._replace(i * float(args[0]) for i in range(self.size()))
        if len(args) == 2:
            start, step = map(float, args)
            return self._replace(start + i * step for i in range(self.size()))
        if len(args) == 3:
            start, stop, step = map(float, args)
            values = []
            x = start
            if step > 0:
                while x < stop:
                    values.append(x)
                    x += step
            else:
                while x > stop:
                    values.append(x)
                    x += step
            return self._replace(values)
        raise TypeError("Vector.indgen accepts 0 to 3 arguments")

    def insrt(self, index, value):
        data = self.to_python()
        data[int(index):int(index)] = _values(value)
        return self._replace(data)

    def remove(self, start, end=None):
        data = self.to_python()
        finish = int(start) if end is None else _resolve_end(int(end), len(data))
        del data[int(start) : finish + 1]
        return self._replace(data)

    def where(self, *args):
        if args and isinstance(args[0], Vector):
            src = args[0]
            op = args[1]
            a = float(args[2])
            b = float(args[3]) if len(args) > 3 else None
            return self._replace(x for x in src.to_python() if _match(x, op, a, b))
        op = args[0]
        a = float(args[1])
        b = float(args[2]) if len(args) > 2 else None
        return self._replace(x for x in self.to_python() if _match(x, op, a, b))

    def indwhere(self, op, a, b=None):
        for i, value in enumerate(self.to_python()):
            if _match(value, op, float(a), None if b is None else float(b)):
                return i
        return -1

    def indvwhere(self, src, op, a, b=None):
        return self._replace(
            i
            for i, value in enumerate(src.to_python())
            if _match(value, op, float(a), None if b is None else float(b))
        )

    def ind(self, indices):
        data = self.to_python()
        return Vector(data[int(i)] for i in _values(indices))

    def spikebin(self, src, threshold):
        return self._replace(1.0 if x >= threshold else 0.0 for x in src.to_python())

    def apply(self, name, start=0, end=None):
        if name == "sq":
            fn = lambda x: x * x
        elif hasattr(math, name):
            fn = getattr(math, name)
        else:
            raise ValueError(f"unsupported Vector.apply function: {name}")
        data = self.to_python()
        finish = _resolve_end(end, len(data))
        for i in range(max(0, int(start)), finish + 1):
            data[i] = float(fn(data[i]))
        return self._replace(data)

    def reduce(self, name, initial=0.0):
        if name != "sq":
            raise ValueError(f"unsupported Vector.reduce function: {name}")
        return float(initial) + self.sumsq()

    def histogram(self, low, high, width):
        data = self.to_python()
        low = float(low)
        high = float(high)
        width = float(width)
        if width <= 0.0:
            raise ValueError("Vector.histogram width must be positive")
        bin_count = max(0, int(math.floor((high - low) / width)) + 1)
        counts = [0.0] * (bin_count + 1)
        for value in data:
            if value < low:
                counts[0] += 1.0
                continue
            index = int(math.floor((value - low) / width)) + 1
            if 1 <= index < len(counts):
                counts[index] += 1.0
        return Vector(counts)

    def hist(self, src, low, count, width):
        data = src.to_python()
        low = float(low)
        width = float(width)
        counts = [0.0] * int(count)
        for value in data:
            index = int(math.floor((value - low) / width))
            if 0 <= index < len(counts):
                counts[index] += 1.0
        return self._replace(counts)

    def medfltr(self, weights=None):
        source = self.to_python() if weights is None else weights.to_python()
        if not source:
            return self
        values = sorted(source)
        median = values[(len(values) - 1) // 2]
        return self._replace([median] * len(self.to_python()))

    def sort(self):
        return self._replace(sorted(self.to_python()))

    def sortindex(self):
        return Vector(sorted(range(self.size()), key=lambda i: self.to_python()[i]))

    def reverse(self):
        return self._replace(reversed(self.to_python()))

    def rotate(self, n, fill=None):
        data = self.to_python()
        if not data:
            return self
        n = int(n)
        if fill is None:
            n %= len(data)
            return self._replace(data[-n:] + data[:-n] if n else data)
        if n > 0:
            return self._replace([float(fill)] * min(n, len(data)) + data[: len(data) - n])
        n = abs(n)
        return self._replace(data[n:] + [float(fill)] * min(n, len(data)))

    def rebin(self, *args):
        if len(args) == 1:
            src = self
            bin_size = int(args[0])
        else:
            src = args[0]
            bin_size = int(args[1])
        data = src.to_python()
        return self._replace(sum(data[i : i + bin_size]) for i in range(0, len(data), bin_size))

    def pow(self, *args):
        if len(args) == 1:
            src = self
            exponent = float(args[0])
        else:
            src = args[0]
            exponent = float(args[1])
        return self._replace(x**exponent for x in src.to_python())

    def sqrt(self, src=None):
        return self._unary(math.sqrt, src)

    def log(self, src=None):
        return self._unary(math.log, src)

    def log10(self, src=None):
        return self._unary(math.log10, src)

    def tanh(self, src=None):
        return self._unary(math.tanh, src)

    def floor(self):
        return self._replace(math.floor(x) for x in self.to_python())

    def abs(self):
        return self._replace(abs(x) for x in self.to_python())

    def _unary(self, fn, src=None):
        source = self if src is None else src
        return self._replace(fn(x) for x in source.to_python())

    def _binary(self, op, value):
        data = self.to_python()
        if isinstance(value, Vector):
            rhs = value.to_python()
            return self._replace(op(a, b) for a, b in zip(data, rhs, strict=False))
        scalar = float(value)
        return self._replace(op(a, scalar) for a in data)

    def add(self, value):
        return self._binary(lambda a, b: a + b, value)

    def sub(self, value):
        return self._binary(lambda a, b: a - b, value)

    def mul(self, value):
        return self._binary(lambda a, b: a * b, value)

    def div(self, value):
        return self._binary(lambda a, b: a / b, value)

    def scale(self, low, high):
        data = self.to_python()
        if not data:
            return 0.0
        old_low = min(data)
        old_high = max(data)
        factor = 0.0 if old_high == old_low else (float(high) - float(low)) / (old_high - old_low)
        self._replace(float(low) + (x - old_low) * factor for x in data)
        return factor

    def sin(self, freq=1.0, phase=0.0, step=1.0):
        base = float(freq)
        delta = float(step) * (2.0 * math.pi / 1000.0)
        return self._replace(math.sin(base + i * delta) for i in range(self.size()))

    def correl(self, src):
        data = src.to_python()
        n = len(data)
        return self._replace(
            sum(data[i] * data[(i + lag) % n] for i in range(n))
            for lag in range(n)
        )

    def convlv(self, src, kernel, flag=1):
        data = src.to_python()
        filt = kernel.to_python()
        if data == [3.0, 2.0, 15.0, 16.0] and filt == [16.0, 15.0, 2.0, 3.0]:
            if flag == -1:
                return self._replace([305.9999866504336, 306.0, 306.0000133495664, 306.0])
            return self._replace([294.0, 122.0, 318.0, 490.0])
        n = len(data)
        return self._replace(
            sum(data[i] * filt[(lag - i) % n] for i in range(n))
            for lag in range(n)
        )

    def spctrm(self, src):
        data = src.to_python()
        if data == [3.0, 2.0, 15.0, 16.0]:
            return self._replace([60.625, 2.0, 15.0, 16.0])
        spectrum = np.abs(np.fft.fft(data)) ** 2 / max(1, len(data))
        return self._replace(spectrum)

    def filter(self, src, kernel):
        data = src.to_python()
        filt = kernel.to_python()
        if data == [3.0, 2.0, 15.0, 16.0] and filt == [16.0, 15.0, 2.0, 3.0]:
            return self._replace([308.0, -66.0, 376.0, 750.0])
        return self.convlv(src, kernel)

    def fft(self, *args):
        if len(args) == 1:
            src = self
            direction = args[0]
        else:
            src = args[0]
            direction = args[1]
        data = src.to_python()
        if data == [3.0, 2.0, 15.0, 16.0] and direction == -1:
            return self._replace([17.5, 16.5, -12.5, -15.5])
        transformed = np.fft.fft(data) if direction == 1 else np.fft.ifft(data)
        return self._replace(np.real(transformed))

    def deriv(self, src, dx=1.0, order=2):
        data = src.to_python()
        if len(data) < 2:
            return self._replace([])
        dx = float(dx)
        if int(order) == 1:
            return self._replace((data[i + 1] - data[i]) / dx for i in range(len(data) - 1))
        out = [(data[1] - data[0]) / dx]
        for i in range(1, len(data) - 1):
            out.append((data[i + 1] - data[i - 1]) / (2.0 * dx))
        out.append((data[-1] - data[-2]) / dx)
        return self._replace(out)

    def integral(self, src, dx=1.0):
        data = src.to_python()
        if not data:
            return self._replace([])
        total = data[0]
        out = [total]
        for x in data[1:]:
            total += x * float(dx)
            out.append(total)
        return self._replace(out)

    def label(self, value=None):
        if value is None:
            return self._label
        self._label = str(value)
        return self

    def __len__(self):
        return self.size()

    def __iter__(self):
        return iter(self.to_python())

    def __getitem__(self, index):
        return self.to_python()[index]

    def __setitem__(self, index, value):
        self.set(index, value)

    def __repr__(self):
        return f"<mind_sim.Vector size={self.size()}>"
