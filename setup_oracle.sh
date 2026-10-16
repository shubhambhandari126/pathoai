#!/usr/bin/env bash
# Pathify one-shot installer for an Oracle Always-Free Ubuntu VM (ARM or x86).
# Usage (on the VM):  curl -sL <this-url> | sudo bash -s -- YOUR_ACCESS_CODE
set -euo pipefail

PASSCODE="${1:?usage: setup_oracle.sh <access-code>}"
APP=/opt/pathify
HF=https://huggingface.co/spaces/Shubhambhandari/Pathify

echo "[pathify] 0/5 checking memory, adding swap on small boxes..."
MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
if [ "${MEM_MB:-0}" -lt 4000 ] && [ ! -f /swapfile ]; then
  echo "[pathify] low RAM (${MEM_MB}MB) -> creating 4G swap"
  fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
# On a small box, skip the demo cohort so boot is light.
if [ "${MEM_MB:-0}" -lt 4000 ]; then CASES=0; SLIDES=0; else CASES=24; SLIDES=3; fi

echo "[pathify] 1/5 installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip python3-venv git curl iptables-persistent

echo "[pathify] 2/5 fetching the app code from Hugging Face..."
rm -rf /tmp/hf_src
git clone --depth 1 "$HF" /tmp/hf_src
mkdir -p "$APP"
cp -r /tmp/hf_src/backend /tmp/hf_src/frontend /tmp/hf_src/pathqa "$APP"/

echo "[pathify] 3/5 downloading the trained model (44 MB)..."
curl -L -o "$APP/model.pt" "$HF/resolve/main/model.pt"

echo "[pathify] 4/5 building python env + torch (several minutes)..."
python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install --upgrade pip -q
"$APP/venv/bin/pip" install -q torch torchvision pillow numpy h5py
mkdir -p "$APP/data/feedback"

echo "[pathify] 5/5 creating the service + opening the port..."
cat >/etc/systemd/system/pathify.service <<EOF
[Unit]
Description=Pathify second-reader QA
After=network.target

[Service]
WorkingDirectory=$APP
Environment=PYTHONPATH=$APP:$APP/backend
Environment=PATHQA_MODEL=$APP/model.pt
Environment=PATHQA_FEEDBACK_DIR=$APP/data/feedback
Environment=PATHQA_AUDIT_LOG=$APP/data/audit.log
Environment=PATHQA_PASSWORD=$PASSCODE
ExecStart=$APP/venv/bin/python $APP/backend/server.py --host 0.0.0.0 --port 80 --cases $CASES --slides $SLIDES --demo-train-steps 0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Ubuntu on Oracle drops inbound traffic except port 22 by default; allow 80.
# Insert at the TOP so it sits before the chain's REJECT rule (else it never applies).
iptables -D INPUT -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT 2>/dev/null || true
iptables -I INPUT 1 -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT || true
netfilter-persistent save || true

systemctl daemon-reload
systemctl enable pathify
systemctl restart pathify

IP=$(curl -s ifconfig.me || echo YOUR_VM_PUBLIC_IP)
echo "=================================================================="
echo " Pathify is starting (first boot loads the model + builds the"
echo " demo worklist, about 1-2 minutes)."
echo ""
echo "   URL:         http://$IP"
echo "   Access code: $PASSCODE"
echo ""
echo " Check status:  sudo systemctl status pathify"
echo " View logs:     sudo journalctl -u pathify -f"
echo "=================================================================="
