# Parallel From Scratch 🚀

This repository is a hands-on study project to **build deep learning parallelism from scratch**.

Inspired by the [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), the goal is not just to *use* distributed training libraries, but to **understand and implement them step by step**.

---

## 🎯 Goal

* Understand how modern large-scale training works
* Implement core parallelism techniques manually
* Share the learning process with others

---

## 🧪 Structure

### Data Parallelism

>  dp1 -> dp2 -> dp3 -> zero1 -> zero2 -> fsdp -> device_mesh

```
data_parallelism/
├── dp1.py
├── dp2.py
├── dp3.py
├── zero1.py
├── zero2.py
├── fsdp.py
├── device_mesh.py
├── dp_benchmark.py
├── utils.py
```

---

## 🚧 Status

Work in progress. Expect incomplete and evolving code.

---

## ⭐ References

* [Ultrascale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
* [PyTorch FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
