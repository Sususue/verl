"""Record bounded intermediate values from the independently executed long and truncated Actor."""
import re
import torch

def install(model,ctx,worker):
    from verl.utils.debug.long_native_edges import install_actor
    install_actor(model,ctx,worker)

    for i,layer in enumerate(model.decoder.layers):
        def later(mod,args,result,i=i):
            case=worker._qualify_case
            if not ctx or not case.startswith('long_') or case.endswith('repeat'):return
            t=result[0] if isinstance(result,(tuple,list)) else result
            if not isinstance(t,torch.Tensor) or t.shape[0]!=ctx['rows']:return
            file=ctx['out']/'positions'/f'L{i:02d}.pt';file.parent.mkdir(exist_ok=True)
            if file.is_file():return
            pos=[p for p in (55,56,63,64,87,95,119,126,127,128,129,130,131,132,135,142,147,159,191,255,283,511) if p<len(t)]
            torch.save(dict(pos=pos,x=t[pos].detach().cpu().contiguous()),file)
        layer.register_forward_hook(later)

    if worker.rank not in (0,1):return
    for name,module in model.named_modules():
        match=re.fullmatch(r'decoder\.layers\.(\d+)\.(self_attention|mlp|mlp\.router|mlp\.shared_experts|self_attention\.linear_q_down_proj|self_attention\.linear_kv_proj)',name)
        if not match:
            match=re.fullmatch(r'decoder\.layers\.([0-5]|8|29)\.(input_layernorm|pre_mlp_layernorm|self_attention\.(?:core_attention|linear_q_up_proj|linear_proj|linear_o_proj|q_layernorm|kv_layernorm|compressor)|mlp\.shared_experts\.linear_fc[12])',name)
        if not match:continue
        label='L'+str(int(match[1])).zfill(2)+'_'+match[2].replace('.','_')
        def observe(mod,args,result,label=label):
            if not ctx or worker._qualify_case not in ('long_first8','long_full','verified64'):return
            folder=ctx['out']/'operators';folder.mkdir(exist_ok=True)
            file=folder/(label+'.pt')
            if file.exists():return
            tensors={}
            for group,parts in [('input',args),('output',result)]:
                if not isinstance(parts,(tuple,list)):parts=(parts,)
                for i,x in enumerate(parts):
                    if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==ctx['rows']:
                        tensors[group+str(i)]=x[:ctx['n']].detach().cpu().contiguous()
            if tensors:torch.save(tensors,file)
        module.register_forward_hook(observe)

