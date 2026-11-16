require "pathname"

root = Pathname("..")/".."
ignored_dirs = %w[
  .devops
  .github
  ci
  examples/wchess/wchess.wasm
  examples/crispasr.android
  examples/crispasr.android.java
  examples/crispasr.objc
  examples/crispasr.swiftui
  grammars
  models
  samples
  scripts
].collect {|dir| root/dir}
ignored_files = %w[
  AUTHORS
  Makefile
  README.md
  README_sycl.md
  .gitignore
  .gitmodules
  .dockerignore
  crispasr.nvim
  twitch.sh
  yt-wsp.sh
  close-issue.yml
  build-xcframework.sh
]

EXTSOURCES =
  # ggml is a git submodule (CrispStrobe/ggml); plain `git ls-files` yields the
  # bare `ggml` gitlink instead of its files, which breaks the `cp` in Rakefile.
  # --recurse-submodules lists the submodule's individual source files.
  `git ls-files --recurse-submodules -z #{root}`.split("\x0")
    .collect {|file| Pathname(file)}
    .reject {|file|
      ignored_dirs.any? {|dir| file.descend.any? {|desc| desc == dir}} ||
        ignored_files.include?(file.basename.to_path) ||
        (file.descend.to_a[1] != root && file.descend.to_a[1] != Pathname("..")/"javascript")
    }
    .collect(&:to_path)
