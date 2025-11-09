# 🔐 Passwords & Credentials Reference

## Quick Access

All passwords and sensitive credentials are now stored in the `.env` file in the project root.

### 📁 Files Created

1. **`.env`** - Contains actual passwords and credentials (gitignored)
2. **`.env.example`** - Template file with placeholder values (safe to commit)
3. **`ENV_SETUP.md`** - Comprehensive setup and usage guide

---

## 🔑 Current Credentials

### Database (PostgreSQL)
- **Host:** localhost
- **Database:** floor_monitor
- **Username:** postgres
- **Password:** `123456` ⚠️ **CHANGE THIS FOR PRODUCTION!**
- **Port:** 5432

### Kaggle API (Optional)
- **Username:** Not configured (update in `.env`)
- **API Key:** Not configured (update in `.env`)
- **Purpose:** Download training datasets from Kaggle

---

## ⚡ Quick Start

### View Current Configuration
```powershell
Get-Content .env
```

### Edit Configuration
```powershell
notepad .env
```

### Backup Configuration
```powershell
Copy-Item .env .env.backup
```

---

## 🛡️ Security Checklist

- [x] `.env` file created with current passwords
- [x] `.env` is in `.gitignore` (prevents accidental commits)
- [x] `.env.example` created as template
- [ ] **TODO: Change default database password**
- [ ] **TODO: Configure Kaggle API if needed**
- [ ] **TODO: Set strong passwords for production**

---

## 📝 Important Notes

1. **Never commit `.env` to Git** - It contains sensitive passwords
2. **Use `.env.example` for sharing** - It has placeholders, not real passwords
3. **Backup your `.env` file** - Store securely outside version control
4. **Rotate passwords regularly** - Update credentials periodically

---

## 🔄 Migration from config.json

The `.env` file has been populated with values from `config/config.json`:

| Setting | config.json | .env |
|---------|-------------|------|
| DB Password | ✓ (visible) | ✓ (hidden from git) |
| DB Host | ✓ | ✓ |
| DB Name | ✓ | ✓ |
| DB User | ✓ | ✓ |
| DB Port | ✓ | ✓ |

**Recommendation:** Keep using `config.json` for non-sensitive settings (camera configs, thresholds, etc.) and `.env` for passwords and API keys.

---

## 🆘 Troubleshooting

### Can't find .env file?
```powershell
# Show hidden files
Get-ChildItem -Force | Where-Object {$_.Name -like ".env*"}
```

### Need to reset .env?
```powershell
# Copy from example
Copy-Item .env.example .env
# Then edit with your actual passwords
```

### Forgot database password?
Check the `.env` file or reset PostgreSQL password:
```sql
ALTER USER postgres WITH PASSWORD 'new_password';
```

---

## 📚 Additional Resources

- **Full Setup Guide:** See `ENV_SETUP.md`
- **Example Template:** See `.env.example`
- **PostgreSQL Docs:** https://www.postgresql.org/docs/
- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api

---

**Last Updated:** 2025-11-09  
**Project:** Floor Monitoring Surveillance System
