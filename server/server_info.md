# Server GPU đích — thông tin kết nối

> **Trạng thái:** Reachable, chưa authorize được. Cần anh Thịnh provide password hoặc paste public key vào `~/.ssh/authorized_keys` qua console nhà cung cấp.

## Thông tin mạng

| Field | Value |
|---|---|
| Host | `89.221.67.144` |
| Port SSH | `22` (default) |
| Reachable | Yes (TCP 22 mở) |
| Connect time | ~10ms (gần) |

## Host key (đã lưu vào `known_hosts`)

| Algorithm | Fingerprint SHA256 |
|---|---|
| RSA-3072 | `lhg7X417vffZsJgUHo7S5RZ7wn00Lr+f+MtVGQLe3qI` |
| ECDSA-P256 | `BlVfxUFqig0SahvkkrOuGlMCCWgUEYLSN/RH8zOHucs` |
| **ED25519** (active) | `Hk0eWOGhyHUBgLINWd+zwnUBxf6RZ2Myjy6MTE/vbJ4` |

**Verify fingerprint:** anh đối chiếu với fingerprint hiển thị trên console nhà cung cấp (Hetzner / OVH / DigitalOcean / …) trước khi tin host key này. Nếu khớp → an toàn. Nếu khác → MITM, dừng kết nối.

## Authentication

| Method | Status |
|---|---|
| publickey | Server accept, **public key của ta CHƯA được authorize** |
| password | Server accept, cần password |
| keyboard-interactive | Không test |

Public key cần paste vào server đích (file `~/.ssh/authorized_keys` của user):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBR40kWh5gpqILG3Eax839cKPHZlu3Igukh6/aeKuxeJ vbk-ai-server-deploy@vietbyte-20260508
```

Fingerprint key của ta: `Sq7p9gbFsA1d3nVW/MQPzAZeyV8neVBBdj4HQagkxAs`

## User đã thử

Em đã thử các user phổ biến đều fail (publickey không authorize, password không có): `root`, `ubuntu`, `deploy`, `admin`, `debian`, `centos`, `azureuser`, `ec2-user`.

## Bước tiếp theo (cần anh quyết)

1. **Cung cấp password** cho user nào (em sẽ paste public key bằng `ssh-copy-id` rồi loại bỏ password auth).
2. **Hoặc** anh tự paste public key trên qua console nhà cung cấp (Hetzner / Vultr / VPS panel).
3. **Hoặc** cho em biết private key đã có sẵn trên máy này (path) và user tương ứng.

## Lệnh deploy 1 cú (sau khi auth xong)

```bash
# Ví dụ user = root, đã authorize key
SERVER=root@89.221.67.144
KEY=/home/thinh/AI_server/server/deploy_key
KNOWN=/home/thinh/AI_server/server/known_hosts

rsync -avz --exclude='.env' --exclude='server/deploy_key' \
    -e "ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN" \
    /home/thinh/AI_server/ "$SERVER:/opt/vbk-ai-server/"

ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN \
    "$SERVER" "cd /opt/vbk-ai-server && cp .env.example .env && \$EDITOR .env && ./scripts/deploy_phase1.sh"
```

## Lưu ý bảo mật

- File `known_hosts` chỉ chứa host key public — commit được.
- Sau khi authorize key, **disable password auth** trên server: `PasswordAuthentication no` trong `/etc/ssh/sshd_config` rồi `systemctl restart ssh`.
- Khuyến nghị dùng `from="<allowlist-IP>",no-port-forwarding,no-agent-forwarding,no-X11-forwarding ssh-ed25519 ...` trong `authorized_keys` của server đích.
