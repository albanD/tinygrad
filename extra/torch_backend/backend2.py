from tinygrad import Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import getenv
import torch, contextlib
from torch.utils._python_dispatch import TorchDispatchMode
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map_only
from functools import wraps
from torch.utils.cpp_extension import load_inline
from torch._decomp import get_decompositions

TORCH_DEBUG = getenv("TORCH_DEBUG")

# Should be in PyTorch core
_old_from_numpy = torch.from_numpy
@wraps(_old_from_numpy)
def _new_from_numpy(array):
  if torch.overrides.has_torch_function_unary(array):
    return torch.overrides.handle_torch_function(_new_from_numpy, (array,), array)
  return _old_from_numpy(array)
torch.from_numpy = _new_from_numpy

src = """
void updateSizeStride(at::Tensor t, int storage_offset, c10::IntArrayRef size, c10::IntArrayRef stride) {
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, storage_offset);
}
"""
_mod = load_inline("_mod", cpp_sources=src, functions=["updateSizeStride"])

# End should be in PyTorch core

torch_to_tiny_dtype = {
  torch.float32: dtypes.float32,
  torch.float64: dtypes.float64,
  torch.int32: dtypes.int32,
  torch.int64: dtypes.int64,
  torch.bool: dtypes.bool,
  torch.uint8: dtypes.uchar,
}
tiny_to_torch_dtype = {v: k for k, v in torch_to_tiny_dtype.items()}

def _to(t, dtype):
  # Is there a function for moving to a given dtype in tinygrad?
  res_np = t.numpy().astype(_to_np_dtype(dtype))
  return Tensor(res_np)
def _broadcast_dims(x, shape, broadcast_dimensions):
  s = list(shape)
  for broadcast_dimension in broadcast_dimensions:
    s[broadcast_dimension] = -1

  v = x.tiny
  for idx, x in enumerate(s):
    if x != -1:
      v = v.unsqueeze(idx)

  return TTensor(v.expand(shape))
def _copy_(x, y):
  x.tiny.assign(y.tiny)
  return x
def _uniform_(x, low, high):
  x.tiny.assign(Tensor.uniform(*tuple(x.shape), low=low, high=high))
  return x
def _normal_(x, mean, std):
  x.tiny.assign(Tensor.normal(*tuple(x.shape), mean=mean, std=std))
  return x
def _fill_scalar_(x, val):
  res = x.tiny.full_like(val)
  x.tiny.replace(res)
  return x
def _set_(x, storage, offset, size, stride):
  new = Tensor(torch.Tensor().set_(storage, offset, size, stride).numpy(force=True))
  # Not cool but we need a way to resize the wrapper inplace...
  x.tiny = new
  _mod.updateSizeStride(x, offset, size, stride)
  return x
def _convolution_(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  return TTensor(input.tiny.conv2d(weight.tiny, bias.tiny if bias is not None else None,
                                   groups=groups, stride=stride, dilation=dilation, padding=padding))
def _maxpool(self:Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: support return_indices in tinygrad
  ret = self.tiny.max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
  # TODO: this is wrong
  return (TTensor(ret), TTensor(Tensor.zeros_like(ret, dtype=dtypes.int64)))

tiny_backend = {
  # "prims.masked_select.default": lambda x,y: TTensor(Tensor(x.tiny.numpy()[y.tiny.numpy()])),
  "aten.view.default": lambda x,sz: TTensor(x.tiny.reshape(sz)),
  "prims.fill.default": lambda x, s: TTensor(Tensor.full(tuple(x.shape), s)),
  "prims.convert_element_type.default": lambda x, dtype: TTensor(_to(x.tiny, torch_to_tiny_dtype[dtype])),
  "prims.sum.default": lambda x, dims: TTensor(x.tiny.sum(dims)),
  "aten._local_scalar_dense.default": lambda x: x.numpy().item(),
  # This one is wrong when dimensions need adding.
  "prims.broadcast_in_dim.default": _broadcast_dims,
  "aten.unsqueeze.default": lambda x,d: TTensor(x.tiny.unsqueeze(d)),
  "aten.permute.default": lambda x,order: TTensor(x.tiny.permute(order)),
  "prims.clone.default": lambda x, **kwargs: TTensor(x.tiny + 0),
  "aten.copy_.default": _copy_,
  "aten.detach.default": lambda x: TTensor(x.tiny.detach()),
  "aten.uniform_.default": _uniform_,
  "aten.normal_.default": _normal_,
  "aten.full_like.default": lambda x, val, **kwargs: TTensor(x.tiny.full_like(val)),
  "aten.fill_.Scalar": _fill_scalar_,
  "aten.set_.source_Storage_storage_offset": _set_,
  "aten.convolution.default": _convolution_,
  "aten.relu.default": lambda x: TTensor(x.tiny.relu()),
  "aten.max_pool2d_with_indices.default": _maxpool,
  "aten.mm.default": lambda x, y: TTensor(x.tiny.matmul(y.tiny)),
}
def unpack(t):
  return t.tiny if isinstance(t, TTensor) else t
def get_fn(fn):
  return lambda x: TTensor(getattr(unpack(x), fn)())
same_names_unary = ["abs", "exp2", "sqrt", "reciprocal"]
for fn in same_names_unary:
  tiny_backend[f"prims.{fn}.default"] = get_fn(fn)

def get_fn2(fn):
  return lambda x, y: TTensor(getattr(unpack(x), fn)(unpack(y)))
same_names_binary = ["eq", "ne", "add", "mul", "sub", "div"]
for fn in same_names_binary:
  tiny_backend[f"prims.{fn}.default"] = get_fn2(fn)

class TTensor(torch.Tensor):
  tiny: Tensor
  context = contextlib.nullcontext

  @staticmethod
  def __new__(cls, tiny, requires_grad=False):
    if not isinstance(tiny, Tensor):
      raise TypeError(f"TTensor constructor expects a tinygrad.Tensor but got {type(tiny)}")
    kwargs = {
      "requires_grad": requires_grad,
      "device": "cpu",
      "dtype": tiny_to_torch_dtype[tiny.dtype],
    }
    out = torch.Tensor._make_wrapper_subclass(cls, tiny.shape, **kwargs)
    torch._C._set_throw_on_mutable_data_ptr(out)
    out.tiny = tiny
    return out
  def __repr__(self): return super().__repr__(tensor_contents=f"{self.tiny}")
  def __torch_dispatch__(cls, func, types, args, kwargs=None):
    if TORCH_DEBUG:
      print(f"DispatchClass: Running on tiny: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
    if func is torch.ops.aten._to_copy.default:
      func = torch.ops.aten.to.dtype
    new_func = tiny_backend.get(str(func), None)
    if new_func is None:
      # Decomp here to allow overriding the decomp from tiny_backend
      if func.namespace != "prims" and (decomp := getattr(torch._refs, func.__name__.split(".")[0], None)) is not None:
        if TORCH_DEBUG:
          print(f"DispatchClass: Decomposing: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
        return decomp(*args, **kwargs)
      else:
        decomps = get_decompositions([func])
        for f, d in decomps.items():
          if TORCH_DEBUG:
            print(f"DispatchClass: Decomposing via aten: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
          return d(*args, **kwargs)
        raise NotImplementedError(f"add support for {func}")
    return new_func(*args, **(kwargs or {}))

  def numpy(self, force=False):
    return self.tiny.numpy()

class FactoryOverride(TorchFunctionMode):
  def __torch_function__(self, func, types, args, kwargs=None):
    kwargs = kwargs or {}

    if "sym" in func.__name__ or \
        any(isinstance(o, torch.Tensor) for o in args) or \
        any(isinstance(o, torch.Tensor) for o in kwargs.values()) or \
        func.__name__ in {"device", "kaiming_uniform_", "_set_grad_enabled"}:
      # Something is a SymInt, Tensor or subclass of Tensor, let C++ handle it
      if TORCH_DEBUG:
        print(f"PythonMode: Running normally: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
      return func(*args, **kwargs)

    # No Tensor input => factory function
    if (decomp := getattr(torch._refs, func.__name__, None)) is not None:
      if TORCH_DEBUG:
        print(f"PythonMode: Decomposing: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
      with self:
        return decomp(*args, **kwargs)

    if func.__name__ == "empty_strided.default":
      if TORCH_DEBUG:
        print(f"PythonMode: Running base factory function: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
      size, stride = args
      running = 1
      for sz, st in zip(reversed(size), reversed(stride)):
        assert st == running, "non-contiguous constructor not done yet"
        running *= sz
      return TTensor(Tensor.empty(*size, dtype=torch_to_tiny_dtype[kwargs["dtype"]]), requires_grad=kwargs.get("requires_grad", False))

    if func.__name__ in {"as_tensor", "from_numpy", "scalar_tensor.default"}:
      if TORCH_DEBUG:
        print(f"PythonMode: Manual handling: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
      if "dtype" in kwargs:
        kwargs["dtype"] = torch_to_tiny_dtype[kwargs["dtype"]]
      if "device" in kwargs:
        # Let tinygrad do autoplacement
        del kwargs["device"]
      return TTensor(Tensor(*args, **kwargs))

    if TORCH_DEBUG:
      print(f"PythonMode: FAILING TO HANDLE: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
    raise NotImplementedError(f"{func.__name__} is a factory function that doesn't decompose to empty_strided?")

FactoryOverride().__enter__()


if __name__ == "__main__":
  a = torch.empty((4,), dtype=torch.int)
