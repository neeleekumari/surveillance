# 🔐 Environment Variables Setup - Quick Start

## ✅ Migration Status: COMPLETE

All passwords have been moved from `config.json` to `.env` file for security.

---

## 🚀 Quick Start

### For New Setup
```bash
# 1. Copy example file
cp .env.example .env

# 2. Edit .env with your passwords
notepad .env

# 3. Run the application
python run.py
```

### For Existing Setup
Your `.env` file is already configured. Just run:
```bash
python run.py
```

---

## 📁 File Structure

```
surveillance/
├── .env                    # ✅ Your passwords (gitignored)
├── .env.example            # Template for new developers
├── config/
│   └── config.json         # ✅ No passwords here anymore
└── src/
    └── config_manager.py   # ✅ Loads passwords from .env
```

---

## 🔑 What's in .env

```bash
# Database credentials
DB_HOST=localhost
DB_NAME=floor_monitor
DB_USER=postgres
DB_PASSWORD=123456          # ⚠️ Change this!
DB_PORT=5432

# Optional: Kaggle API
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

---

## ✅ Verification

Run the test suite to verify everything works:
```bash
python test_env_migration.py
```

Expected output:
```
🎉 ALL TESTS PASSED! Migration successful!
```

---

## 🛡️ Security Benefits

| Before | After |
|--------|-------|
| ❌ Password in `config.json` | ✅ Password in `.env` |
| ❌ Committed to Git | ✅ Gitignored |
| ❌ Visible to everyone | ✅ Secure |

---

## 📚 Documentation

- **Quick Reference:** `PASSWORDS_REFERENCE.md`
- **Full Setup Guide:** `ENV_SETUP.md`
- **Migration Details:** `MIGRATION_COMPLETE.md`

---

## ⚠️ Important

1. **Never commit `.env` to Git** - It's already gitignored
2. **Change default password** - Update `DB_PASSWORD` in `.env`
3. **Backup `.env` file** - Store securely outside Git

---

## 🆘 Troubleshooting

### Database connection fails
```bash
# Check .env file exists
ls -la .env

# Verify password is set
grep DB_PASSWORD .env
```

### Password not loading
```bash
# Reinstall python-dotenv
pip install --upgrade python-dotenv

# Run test
python test_env_migration.py
```

---

## 🎯 Next Steps

1. ✅ Migration complete
2. ⚠️ **Change default password in `.env`**
3. ⚠️ **Update PostgreSQL password to match**
4. ✅ Continue development as normal

---

**Status:** ✅ Ready to use  
**Last Updated:** 2025-11-09
