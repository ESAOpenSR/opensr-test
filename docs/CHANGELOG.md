# Changelog

All notable changes to the `opensr-test` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Added
- For new features.
### Changed
- For changes in existing functionality.
### Deprecated
- For soon-to-be removed features.
### Removed
- For now removed features.
### Fixed
- For any bug fixes.
### Security
- In case of vulnerabilities.

## [Unreleased]

Unreleased changes here.

## [1.3.2] - 2024-10-31

- We change the CLIP algorithm from `satclip` to generic `clip` function.

## [1.2.2] - 2024-10-31

- A better algorithm to Harmonize the LR and HR images.

## [1.2.1] - 2024-09-02

- Bug due to the new satalign package API fixed.

## [1.2.0] - 2024-06-10

- Fractional Difference Index (FDI) key is now both `fd` and `nd`. In order to
avoid confusion, with Normalized Difference (NDI). They are in essence the
same index. This is the metric by default from now on.
- We change the .png by .gif in the README.md file.
- We update the `README.md` benchmark section, with the new results (SUPER-IX feedback).
- We add a colab badge in the `README.md` file.

## [1.1.0] - 2024-05-20

- Fractional Difference Index (FDI) added to the distance metrics.
- Bug in the `plot_tc` function fixed.
- A new parameter added to the `display_ternary` function to control the bin count.
- Config initial parameters have been updated.



## [1.0.0] - 2024-05-20

- Logo changed.
- The `lightglue` submodule has been replaced with a new package called `satalign`.
- A new example has been added on how to run `opensr-test` on synthetic data.
- The documentation has been updated.
- The harmonization module has been updated to include the `satalign` package.
- We have added two new datasets: `spain_crops` and `spain_urban`.
- We add a new plot function: `plot_ternary`.
- We add a new plot function: `plot_histogram`.




## [0.2.0] - 2023-12-20

- Paper submited to the IEEE remote sensing letters.


## [0.1.0] - 2023-12-20

### Added
- First release of the `opensr-test` package.


[Unreleased]: https://github.com/ESAOpenSR/opensr-test/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ESAOpenSR/opensr-test/releases/tag/v0.1.0
