# Releasing HNSWLib

## Prerequisites

- Hex.pm account with publish access to `hnswlib`
- All tests passing on main
- Repository secrets configured: `RELEASE_PAT`, `HEX_API_KEY`

## Release Process

Releases are automated via [release-please](https://github.com/googleapis/release-please).

### How it works

1. **Push to main** - release-please analyzes commits and creates/updates a release PR
2. **Review PR** - The PR includes version bump and CHANGELOG updates
3. **Merge PR** - Merging the PR (labeled `autorelease: pending`) triggers:
   - Tag creation
   - Build of all platform binaries
   - GitHub release with artifacts
   - Checksum generation and commit
   - Hex.pm publish

### Commit message format

release-please uses [Conventional Commits](https://www.conventionalcommits.org/) to determine version bumps:

- `fix:` - Patch release (0.0.X)
- `feat:` - Minor release (0.X.0)
- `feat!:` or `BREAKING CHANGE:` - Major release (X.0.0)

### Release flow

```
push to main (with conventional commits)
    ↓
release-please creates/updates PR
    ↓
merge PR (labeled "autorelease: pending")
    ↓
create-release.yml creates tag
    ↓
build-and-publish.yml triggered by tag
    ↓
build artifacts → GitHub release → checksum commit → hex publish
```

## Manual Release (Emergency)

If automated release fails, you can manually:

1. Create and push a tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

2. This triggers `build-and-publish.yml` which handles the rest.

3. If hex publish fails, manually publish:
   ```bash
   git checkout vX.Y.Z
   mix deps.get
   mix hex.publish
   ```
