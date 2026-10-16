#!/usr/bin/env bash
# Put Pathify behind a domain with automatic free HTTPS (Caddy + Let's Encrypt).
# Run AFTER the domain's DNS already points to this VM's public IP.
# Usage (on the VM):  curl -sL <this-url> | sudo bash -s -- your.domain
set -euo pipefail
DOMAIN="${1:?usage: setup_domain.sh <domain>}"

echo "[pathify] moving the app to internal port 8080..."
sed -i 's/--port 80 /--port 8080 /' /etc/systemd/system/pathify.service
systemctl daemon-reload
systemctl restart pathify

echo "[pathify] opening firewall for 80 and 443..."
for p in 80 443; do
  iptables -D INPUT -p tcp -m state --state NEW -m tcp --dport "$p" -j ACCEPT 2>/dev/null || true
  iptables -I INPUT 1 -p tcp -m state --state NEW -m tcp --dport "$p" -j ACCEPT
done
netfilter-persistent save || true

echo "[pathify] installing Caddy (auto-HTTPS reverse proxy)..."
export DEBIAN_FRONTEND=noninteractive
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
apt-get update -y >/dev/null
apt-get install -y caddy >/dev/null

echo "[pathify] configuring HTTPS for $DOMAIN..."
cat >/etc/caddy/Caddyfile <<EOF
$DOMAIN {
    reverse_proxy 127.0.0.1:8080
}
EOF
systemctl restart caddy

echo "=================================================================="
echo " Done. Caddy is fetching a free certificate now (~30-60 seconds)."
echo " Then open:  https://$DOMAIN"
echo " Check Caddy: sudo journalctl -u caddy --no-pager | tail -20"
echo "=================================================================="
