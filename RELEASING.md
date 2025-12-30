# Releasing HNSWLib

## Prerequisites

- Hex.pm account with publish access to `hnswlib`
- All tests passing on main
- `HEX_API_KEY` secret configured in GitHub repository settings (for automated release)

## Automated Release (Recommended)

1. Ensure `mix.exs` has a version ending with `-dev` (e.g., `0.1.7-dev`)
2. Go to [Actions > Release](https://github.com/elixir-nx/hnswlib/actions/workflows/release.yml)
3. Click "Run workflow"
4. The workflow will automatically:
   - Update version to release (strip `-dev`)
   - Wait for tests to pass
   - Create and push tag
   - Wait for precompile to build all binaries
   - Generate `checksum.exs`
   - Publish to Hex
   - Bump to next `-dev` version

## Manual Release

### Steps

### 1. Update version for release

Edit `mix.exs` and change `@version "X.Y.Z-dev"` to `"X.Y.Z"`.

### 2. Commit and push

```bash
git add mix.exs
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### 3. Wait for CI

- Tests must pass
- Precompile builds all binaries (triggered after tests pass)
- Monitor at: https://github.com/elixir-nx/hnswlib/actions

### 4. Create and push tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### 5. Wait for release CI

- Precompile workflow builds all platform binaries
- Creates GitHub release with all `.tar.gz` and `.sha256` files
- Monitor at: https://github.com/elixir-nx/hnswlib/actions

### 6. Generate checksum.exs

```bash
mix elixir_make.checksum --all --ignore-unavailable
```

### 7. Commit checksum

```bash
git add checksum.exs
git commit -m "Add checksum.exs for vX.Y.Z"
git push origin main
```

### 8. Publish to Hex

```bash
git checkout vX.Y.Z
mix deps.get
mix hex.publish
```

### 9. Bump to next dev version

```bash
git checkout main
# Edit mix.exs: change @version "X.Y.Z" to "X.Y.(Z+1)-dev"
git add mix.exs
git commit -m "Bump version to X.Y.(Z+1)-dev"
git push origin main
```

## Summary

```
main (X.Y.Z-dev) → bump to X.Y.Z → push → CI passes → tag vX.Y.Z →
CI builds release → checksum.exs → hex.publish → bump to X.Y.(Z+1)-dev
```
