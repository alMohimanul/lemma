# Open Source Release Checklist

## ✅ Pre-Release Checklist

### Documentation
- [x] README.md with clear description and quick start
- [x] CONTRIBUTING.md with contribution guidelines
- [x] LICENSE file (MIT License)
- [x] Clear installation instructions
- [x] Usage examples in README
- [ ] CHANGELOG.md for version 2.0.0
- [ ] API documentation (if needed)
- [ ] Screenshots/GIFs for README

### Code Quality
- [x] Pre-commit hooks configured
- [x] Linting setup (ruff)
- [x] Type checking (mypy)
- [x] Tests written (pytest)
- [ ] Test coverage >70%
- [ ] Code formatted consistently
- [ ] No hardcoded secrets or API keys
- [ ] All TODO comments resolved or tracked

### Repository Setup
- [x] .gitignore properly configured
- [x] No sensitive data in git history
- [x] Clean git history
- [x] Feature branches merged
- [ ] Main branch protected
- [ ] Branch protection rules set

### CI/CD
- [x] GitHub Actions workflows
- [x] Test automation
- [x] Build automation
- [ ] Release automation tested
- [ ] PyPI credentials configured

### Package Management
- [x] setup.py with complete metadata
- [x] requirements.txt
- [x] Version number updated (2.0.0)
- [ ] Package tested locally (`pip install -e .`)
- [ ] Package built successfully (`python -m build`)
- [ ] Package uploaded to TestPyPI (test first)

### Security
- [ ] Security policy (SECURITY.md)
- [ ] Dependency vulnerability scan
- [ ] No known security issues
- [ ] API keys handled securely (.env pattern)

### Legal
- [x] MIT License in place
- [x] Copyright year correct (2026)
- [ ] Third-party licenses acknowledged
- [ ] Attribution for borrowed code

### Community
- [ ] Issue templates
- [ ] PR template
- [ ] Code of Conduct
- [ ] Discussion forum enabled
- [ ] Social media ready announcement

---

## 🚀 Release Process

### 1. Final Testing
```bash
# Clean install test
python -m venv test-venv
source test-venv/bin/activate
pip install -e .
lemma --help

# Run full test suite
pytest tests/ -v --cov=src

# Test on fresh database
rm -rf ~/.lemma/
lemma sync ~/test-papers
```

### 2. Update Version & Changelog
```bash
# Update version in setup.py to 2.0.0
# Create/update CHANGELOG.md
# Commit changes
git add setup.py CHANGELOG.md
git commit -m "chore: Bump version to 2.0.0"
```

### 3. Create Release Branch
```bash
git checkout -b release/v2.0.0
git push origin release/v2.0.0
```

### 4. Create GitHub Release
- Tag: `v2.0.0`
- Title: `Lemma v2.0.0 - Paper Comparison & Enhanced Features`
- Description: Include highlights from CHANGELOG
- Attach build artifacts

### 5. Publish to PyPI
```bash
# Build package
python -m build

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install -i https://test.pypi.org/simple/ lemma-ai

# If OK, upload to real PyPI
twine upload dist/*
```

### 6. Post-Release
- [ ] Announce on Twitter/LinkedIn
- [ ] Post on Reddit (r/Python, r/MachineLearning)
- [ ] Submit to awesome lists
- [ ] Update project website (if exists)
- [ ] Monitor issues/discussions

---

## 📝 Recommended Additions

### Must Have Before v2.0.0 Release
1. **CHANGELOG.md** - Document all changes since v1.0
2. **Issue Templates** - Bug report, feature request
3. **Security Policy** - How to report vulnerabilities
4. **Test Coverage Report** - Badge in README

### Nice to Have
1. **Code of Conduct** - Contributor Covenant
2. **Screenshots** - Show terminal output
3. **Demo Video** - Quick walkthrough
4. **Documentation Site** - Using ReadTheDocs or similar
5. **Badges** - CI status, coverage, PyPI version

### Social Media Assets
1. **Twitter Thread** - Feature highlights
2. **LinkedIn Post** - Professional announcement
3. **Dev.to Article** - Technical deep dive
4. **Demo GIF** - Show paper comparison in action

---

## 🎯 Quick Wins for Visibility

### GitHub
- [ ] Add topics/tags to repository
- [ ] Enable GitHub Sponsors (optional)
- [ ] Pin important issues
- [ ] Create GitHub Project for roadmap

### Community
- [ ] Submit to Papers with Code
- [ ] Post on Hacker News Show HN
- [ ] Share in academic Twitter
- [ ] Join relevant Discord/Slack communities

### SEO
- [ ] Good README with keywords
- [ ] Detailed PyPI description
- [ ] Links from personal site/blog
- [ ] Medium article with tutorial

---

## ⚠️ Critical Issues to Fix

### Before Going Public
1. **Remove .env from repository** - Already in .gitignore, verify not committed
2. **Check git history** - No API keys or secrets in old commits
3. **Test on fresh clone** - Ensure everything works from scratch
4. **Verify all links** - GitHub URLs, documentation links
5. **Clean up branches** - Delete obsolete feature branches

### Current Status
- ✅ Feature branch: `feat/paper-comparison` ready to merge
- ⚠️ Need to update version to 2.0.0
- ⚠️ Need CHANGELOG.md
- ⚠️ Should add issue templates
