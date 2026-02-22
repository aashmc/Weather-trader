#!/bin/bash
# ══════════════════════════════════════════════════════
# Weather Trader Bot — Hetzner Deploy Script
# Run this ONCE on a fresh Ubuntu 24.04 VPS
# Usage: bash deploy.sh
# ══════════════════════════════════════════════════════

set -e

echo "══════════════════════════════════════════"
echo "  Weather Trader Bot — Deploying"
echo "══════════════════════════════════════════"

# 1. System updates
echo "→ Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

# 2. Create bot directory
BOT_DIR=/opt/weather-trader
echo "→ Setting up $BOT_DIR..."
sudo mkdir -p $BOT_DIR
sudo chown $USER:$USER $BOT_DIR

# 3. Copy files (run this from the repo directory)
cp bot.py ensemble.py market.py metar.py strategy.py risk.py alerts.py logger.py config.py requirements.txt $BOT_DIR/

# 4. Create virtual environment
echo "→ Creating Python venv..."
cd $BOT_DIR
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
echo "→ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Create .env (if not exists)
if [ ! -f "$BOT_DIR/.env" ]; then
    cp /opt/weather-trader/.env.example $BOT_DIR/.env 2>/dev/null || cat > $BOT_DIR/.env << 'ENVEOF'
POLYGON_PRIVATE_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
GOOGLE_SHEET_WEBHOOK=
ENVEOF
    echo ""
    echo "⚠️  IMPORTANT: Edit $BOT_DIR/.env with your secrets:"
    echo "    sudo nano $BOT_DIR/.env"
    echo ""
fi

# 7. Create systemd service
echo "→ Creating systemd service..."
sudo tee /etc/systemd/system/weather-trader.service > /dev/null << EOF
[Unit]
Description=Weather Trader Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BOT_DIR
ExecStart=$BOT_DIR/venv/bin/python3 $BOT_DIR/bot.py
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

# Auto-restart on crash, max 5 restarts in 5 min
StartLimitBurst=5
StartLimitIntervalSec=300

[Install]
WantedBy=multi-user.target
EOF

# 8. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable weather-trader
sudo systemctl start weather-trader

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ DEPLOYED SUCCESSFULLY"
echo "══════════════════════════════════════════"
echo ""
echo "  Useful commands:"
echo "    View logs:      sudo journalctl -u weather-trader -f"
echo "    Stop bot:       sudo systemctl stop weather-trader"
echo "    Start bot:      sudo systemctl start weather-trader"
echo "    Restart bot:    sudo systemctl restart weather-trader"
echo "    Bot status:     sudo systemctl status weather-trader"
echo "    Edit secrets:   sudo nano $BOT_DIR/.env"
echo ""
echo "  ⚠️  Don't forget to:"
echo "    1. Edit .env with your secrets"
echo "    2. Restart after editing: sudo systemctl restart weather-trader"
echo ""
