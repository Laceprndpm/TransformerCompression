rsync -avz \
  -e "ssh -p 10197" \
  --exclude-from='.rsyncignore' \
  /home/patchouli/Projects/TransformerCompression/ \
  root@connect.cqa1.seetacloud.com:/root/autodl-tmp/exp/TransformerCompression/