# medpipe

# Run

- search
```bash
bash search.sh -a vit_b --search
```

- finetune
```bash
bash finetune.sh -a vit_b --resume /path/to/name.pt --arch_path /path/to/arch_name.json 
```

- evaluate
```bash
bash evaluate.sh -a vit_b --arch_path /path/to/arch_name.json --resume /path/to/name.pt --evaluate
```
