# Instructions to create a new dvas release

These steps are valid as long as we remain in the *private* development phase.

1. Bump the code version by changing `./src/dvas/version.py`, and update the `CHANGELOG` by
   separating the latest changes into the corresponding version.

2. Send both these changes to the `develop` branch with a Pull Request.
   Make sure all the tests are green before merging.

3. From Github, create a Pull Request from `develop` to `master`.

4. If all is green, merge the Pull Request into the Github `master`. The docs will be automatically
   updated on the `gh-pages` branch.

## Steps to automate down the line (i.e. once we enter the public phase)
See [issue #82](https://github.com/MCH-MDA/dvas/issues/82) for an up-to-date todo list.
