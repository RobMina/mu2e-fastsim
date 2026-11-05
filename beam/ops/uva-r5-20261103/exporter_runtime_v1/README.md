# Immutable plugin deployment candidates

These two `.plugin` files are unchanged user-built ELF plugins, not a build or
muse tarball. Their suffix permits normal mirrored delivery without changing
shared binary ignores. A runtime probe copies them into an ephemeral, explicitly
pinned satellite area's `build/al9-prof-e29-p109/FlashReference/lib` with their
normal art plugin names, and checks hashes before loading. No build is invoked.

- `FlashReferenceExporter.plugin`: `bf61d4b5fafc6ac52119059f08171e943384da81a3aaba28094a73bc57284189`
- `FlashReferenceFixture.plugin`: `1ec75a380b4714fe89827ccdea52596cb83d70827c3bee5def242fdde7df8c15`

Backing: Offline v13_41_01, envset p109, art 3.15, ROOT 6.38. Source hashes and
user-build verification are retained in the R5 live record. This directory alone
is not evidence of successful loading or extraction on UVA. No transport launch
or aggregate-output guarantee follows from deployment.
