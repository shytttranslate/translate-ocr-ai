# server/ — SSH key + deploy artifact cho server GPU

Folder này chứa SSH key dùng để deploy stack lên server có GPU NVIDIA.

## File

| File | Mục đích | Commit? |
|---|---|---|
| `deploy_key` | Private key ed25519 (600) | KHÔNG — đã trong `.gitignore` |
| `deploy_key.pub` | Public key, cần paste vào server đích | Có thể commit |
| `.gitignore` | Loại trừ private key khỏi git | Có |

## Cách dùng

### 1. Cài public key lên server GPU đích

Trên máy hiện tại:
```bash
cat /home/thinh/AI_server/server/deploy_key.pub
```

Paste output vào file `~/.ssh/authorized_keys` của user trên server GPU (ví dụ user `deploy`):

```bash
# Chạy trên server GPU đích
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-ed25519 AAAA... vbk-ai-server-deploy@..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 2. Test kết nối

```bash
ssh -i /home/thinh/AI_server/server/deploy_key -o IdentitiesOnly=yes \
    deploy@<gpu-server-ip> "nvidia-smi"
```

### 3. Rsync project lên server

```bash
rsync -avz --exclude='.env' --exclude='server/deploy_key' \
    -e "ssh -i /home/thinh/AI_server/server/deploy_key -o IdentitiesOnly=yes" \
    /home/thinh/AI_server/ \
    deploy@<gpu-server-ip>:/opt/vbk-ai-server/
```

### 4. Deploy từ xa

```bash
ssh -i /home/thinh/AI_server/server/deploy_key deploy@<gpu-server-ip> \
    "cd /opt/vbk-ai-server && ./scripts/deploy_phase1.sh"
```

## Bảo mật

- Private key KHÔNG passphrase để automation chạy được. Đổi lại: phải bảo vệ file `deploy_key` chặt chẽ — chỉ user sở hữu đọc được (mode 600), folder `server/` mode 700.
- Nếu nghi ngờ key bị lộ: xoá public key khỏi `~/.ssh/authorized_keys` server đích, sinh lại key mới.
- Nên hạn chế quyền key này trên server đích bằng `command=` trong `authorized_keys`:
  ```
  command="cd /opt/vbk-ai-server && ./scripts/deploy_phase1.sh",no-port-forwarding,no-X11-forwarding ssh-ed25519 AAAA...
  ```
- Khuyến nghị: thêm `from="<ip-cidr-allowlist>"` trước key để chỉ accept từ IP cố định.

## Sinh lại key (nếu cần)

```bash
rm /home/thinh/AI_server/server/deploy_key{,.pub}
ssh-keygen -t ed25519 \
    -f /home/thinh/AI_server/server/deploy_key \
    -N "" \
    -C "vbk-ai-server-deploy@$(hostname)-$(date +%Y%m%d)"
chmod 600 /home/thinh/AI_server/server/deploy_key
chmod 644 /home/thinh/AI_server/server/deploy_key.pub
```
