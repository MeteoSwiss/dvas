# Contributing to dvas

If you want to report a bug with dvas, [jump here](#reporting-a-bug).

If you are still reading this, you may actually be considering contributing to the development of dvas. :heart_eyes: :tada:

There are many ways that you can do so, including by:
- [reporting a bug](#reporting-a-bug)
- fixing an [known issue](https://github.com/MeteoSwiss-MDA/dvas/issues?q=is%3Aissue+),
- implementing a new functionality, and/or
- improving the documentation:
  * in the code, with better docstrings
  * in this repository (for example this very file !)
  * in the website, via the docs `.rst` files

All these contributions are welcome, and what follows should help you get started. Note that contributing to dvas does *not* necessarily require an advanced knowledge of Python and/or Github. Helping us fix typos in the docs, for example, could be an excellent first contribution. Plus, :anger: typos :anger: are the worst !

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Reporting a bug](#reporting-a-bug)
- [Essential things to know about dvas](#essential-things-to-know-about-dvas)
- [Styles](#styles)
- [Step-by-step guide to contributing](#step-by-step-guide-to-contributing)

## Code of conduct

This project and everyone participating in it is governed by the [dvas Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

## Reporting a bug

If you find something odd with dvas, first check if it is a [known issue](https://github.com/MeteoSwiss-MDA/dvas/issues?q=is%3Aissue+). If not, please create a new [Github Issue](https://github.com/MeteoSwiss-MDA/dvas/issues). This is the best way for everyone to keep track of new problems and past solutions.

## Essential things to know about dvas
dvas is a Python module. But dvas also includes a series of parameter and
utilitarian files related to its Github repository, and a dedicated documentation hosted using Github pages.

For the sake of clarity, and to facilitate the maintenance, we list here (succinctly) a series of key facts about the dvas code and its repository:

1. **Source code:**
   * dvas is distributed under the terms of the GNU General Public License v3.0 or later. The dvas
    copyright is owned by MeteoSwiss, with the following [authors](AUTHORS).
   * dvas adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
   * The adopted styles are described [here](#styles).
   * dvas *operational* dependencies are specified in `setup.py`.
   * There is a human-readable [Changelog](CHANGELOG).

2. **Github repository:**
   * Contributions to dvas get typically merged into the `develop` branch. Pull requests to the
     `master` branch should only originate from the `develop` branch.
   * Any successful pull request to the `master` branch should trigger a new code release.
   * A series of Github Actions are implemented for CI purposes. These include the execution of
     the dvas tests on Windows, macOS and Linux, a linting of the code, a validation
     of the docs, and a check of the `CHANGELOG`. Upon any push to the `master` branch, the docs
     will also be automatically compiled and published onto the `gh-pages` branch.
   * The `.pylintrc` file refines the behavior of pylint for dvas.

3. **Documentation:**
   * The dvas documentation is generated using Sphinx, with the Read-the-docs theme. The compiled
     documentation is hosted on the `gh-pages` branch of the dvas repository.
   * UML diagrams included in the code docstrings are rendered (when building the docs) with the
     [plantuml server](http://www.plantuml.com/plantuml).

4. **Development utilities:**
   * Dependencies required for *code development* activities are specified in
     `./.dev_utils/dev_requirements.txt`.
   * On Windows, linter and tests can be run locally from terminal with `sh .\.dev_utils\linter_bash.bat`
     resp. `sh .\.dev_utils\test_bash.bat` commands.

## Styles

- **linting:**
  * The following [pylint](https://www.pylint.org/) error codes are forbidden in dvas: ``E, C0303, C0304, C0112, C0114, C0115, C0116, C0411, W0611, W0612.`` Any pull request will be automatically linted, and these will be flagged accordingly.
  * We encourage contributors to follow PEP8 as closely as possible/reasonable. You should check
    often how well you are doing using the command `pylint some_modified_file.py`.
  * To avoid `E1101` errors, stick to the following:
    ```
    import netCDF4 as nc
    ```

- **doctrings:** Google Style. Please try to stick to the following MWE:
```
    """ A brief one-liner description, that finishes with a dot.

    Use some
    multi-line space for
    more detailed info.

    Args:
        x (float, int): variable x could be of 2 types ...

           - *float*: x could be a float
           - *int*: x could also be an int

        y (list[str]; optional): variable y info

    Returns:
        bool: some grand Truth about the World.

    Example:
        If needed, you can specify chunks of code using code blocks::

            def some_function():
                print('hurray!')

    Note:
        `Source <https://github.com/sphinx-doc/sphinx/issues/3921>`__
        Please note the double _ _ after the link !

    Caution:
        Something the be careful about.

    .. uml::
        @startuml
        title Sequence diagram example
        Alice -> Bob: Hi!
        Alice <- Bob: How are you?
        @enduml

    """
```
You should of course feel free to use more of the tools offered by [sphinx](https://www.sphinx-doc.org/en/master/), [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html), and
[Google Doc Strings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google). But if you do, **please make sure there are no errors upon generating the docs !**

## Step-by-step guide to contributing

We are currently in the early stages of development of dvas. If you would like to contribute to the code, please contact [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

Until its release, the dvas repository will remain private: branching will thus remain the only way to contribute to the code. To get a local copy of dvas and contribute to its improvement, one can follow the following steps:

0. Make sure you have git installed. Check that the setup is correct:

       git config --list

   If `user.name` and `user.email` are missing or do not match those of your Github account account, change them:

       git config --local user.name "your_github_id"
       git config --local user.email "your_github_id@users.noreply.github.com"

1. Clone the develop branch locally:

       git clone -b develop https://github.com/MeteoSwiss-MDA/dvas.git your_branch_name

2. Actually create your new branch locally:

       cd your_branch_name
       git checkout -b your_branch_name

3. Check that it all went as expected:

       git branch -a
       git config --list
       git status

4. Install the packages that are required for doing dev work with dvas:

       pip install -r dev_requirements.txt

5. Modify the code locally. This could be the source code, or the docs `.rst` source files.

   :warning: Please read carefully (and adhere to!) the dvas [style conventions](#styles) below.

6. Commit changes regularly, trying to bundle them in a meaningful manner.

       git add a_modified_file (OR possibly: git rm a_file_to_delete)
       git commit -m "Some useful, clear, and concise message. Use present tense."

   You can/should also push your branch to the dvas repository, if you want others to see what you are up to:

       git push origin your_branch_name

7. Lint your contributions using the command `pylint some_modified_file.py`. If you want to run the
   checks that will be executed automatically at the pull request stage, you can run the following
   commands from the dvas repository:

       python ./.github/workflows/pylinter.py --restrict E C0303 C0304 C0112 C0114 C0115 C0116 C0411 W0611 W0612
       python ./.github/workflows/pylinter.py --min_score 8

    Note that this may pick-up linting problems outside of your contribution as well.

8. If warranted, make sure that the docs still compile without errors/warnings:

       cd docs
       sh build_docs.sh

9. Once ready with all your modifications, we'll ask that you do a rebase of your branch to
           incorporate any modification that may have occurred on the original `develop` branch in the meantime:

              git fetch origin develop
              git pull --rebase origin develop

10. You can now push your branch to the dvas repository. If warranted (it most likely will be!),
    remember to update the `CHANGELOG` and add your name to the `AUTHORS` before doing so.

       git push -f origin your_branch_name

    Note the `-f` flag, required because of the `--rebase` to update the commit history of the
    branch stored on Github.

11. Next, go to `your_branch_name` on the dvas Github repository, and draft a new pull request. By
    default, the pull request should go from `your_branch_name` to the `develop` branch. Do not forget to link the pull request to a specific issue if warranted. Once the pull request is issued, automated checks will be run (pytest, pylint, changelog, ...). These must all succeed before the changes can be merged (if they do not, there might be something wrong with your changes).

    The code devs will then come to formally review the pull request (some reviewers might also be
    set automatically based on the `.github/CODEOWNERS` info).
