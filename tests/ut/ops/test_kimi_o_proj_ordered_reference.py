"""CPU contract for the Kimi O-projection-only ordered BF16 reference.

Uses the production method AST without importing the full model or claiming NPU/model acceptance.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def _source():
    return Path(__file__).resolve().parents[3]/'vllm_ascend/ops/linear_op.py'


def _class_and_method():
    tree=ast.parse(_source().read_text(encoding='utf8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='KimiOProjMMReduceScatterOp')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_ordered_sp_reduce_scatter')
    return cls,method


class FakeDist:
    def all_gather_into_tensor(self,out,partial,group):
        size=len(group.partials)
        assert size==4 and partial.shape[0]%size==0
        padded=[]
        for row in group.partials:
            pad=(-row.shape[0])%size
            padded.append(torch.nn.functional.pad(row,(0,0,0,pad)) if pad else row)
        assert torch.equal(partial,padded[group.rank])
        out.copy_(torch.cat(padded,dim=0))


def _invoke(partials,rank):
    _,method=_class_and_method()
    namespace={'torch':torch,'dist':FakeDist()}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(_source()),'exec'),namespace)
    group=SimpleNamespace(partials=partials,rank=rank)
    op=SimpleNamespace(tp_size=len(partials),tp_rank=rank,comm_group=SimpleNamespace(device_group=group))
    return namespace['_ordered_sp_reduce_scatter'](op,partials[rank])


def test_same_prompt_changes_owner_without_changing_reduction():
    vectors=[torch.tensor([v,-v,0.5],dtype=torch.bfloat16) for v in (0.25,0.125,-0.5,0.0625)]
    results=[]
    for rows,token_row in ((16,8),(2,1)):
        partials=[torch.zeros((rows,3),dtype=torch.bfloat16) for _ in range(4)]
        for rank,vector in enumerate(vectors):partials[rank][token_row]=vector
        owner=token_row//((rows+3)//4)
        out=_invoke(partials,owner)
        assert out.shape==((rows+3)//4,3) and out.dtype==torch.bfloat16
        results.append(out[token_row%out.shape[0]].clone())
    expected=vectors[0].clone()
    for v in vectors[1:]:expected.add_(v)
    assert torch.equal(results[0],results[1]) and torch.equal(results[0],expected)


def test_aligned_and_padded_paths_preserve_input_and_no_alias():
    for rows in (2,16):
        partials=[torch.ones((rows,4),dtype=torch.bfloat16)*(rank+1) for rank in range(4)]
        original=[x.clone() for x in partials]
        for rank in range(4):
            result=_invoke(partials,rank)
            chunk=(rows+3)//4
            expected=torch.zeros((chunk,4),dtype=torch.bfloat16)
            live_rows=max(0,min(chunk,rows-rank*chunk))
            expected[:live_rows].fill_(10)
            assert result.shape==(chunk,4) and torch.equal(result,expected)
            result.zero_()
            assert all(torch.equal(x,y) for x,y in zip(partials,original))


def test_wrong_dtype_or_rank_is_rejected():
    partials=[torch.ones((2,4),dtype=torch.float32) for _ in range(4)]
    with pytest.raises(AssertionError):_invoke(partials,0)
    partials=[torch.ones((2,),dtype=torch.bfloat16) for _ in range(4)]
    with pytest.raises(AssertionError):_invoke(partials,0)


def test_only_nonfused_fallback_uses_reference():
    cls,_=_class_and_method()
    apply=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='apply_impl')
    text=ast.get_source_segment(_source().read_text(encoding='utf8'),apply)
    assert 'if not self._can_fuse(input_parallel, weight):' in text
    assert 'return self._ordered_sp_reduce_scatter(output), None' in text
    assert 'torch_npu.npu_quant_mm_reduce_scatter(' in text
