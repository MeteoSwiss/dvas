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

All these contrbutions are welcome, and what follows should help you get started. Note that contributing to dvas does *not* necessarily require an advanced knowledge of python and/or Github. Helping us fix typos in the docs, for example, could be an excellent first contribution. Plus, :anger: typos :anger: are the worst !

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Reporting a bug](#reporting-a-bug)
- [Contributing](#contributing)
- [Styles](#styles)

## Code of conduct

This project and everyone participating in it is governed by the [dvas Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

## Reporting a bug

If you find something odd with dvas, first check if it is a [known issue](https://github.com/MeteoSwiss-MDA/dvas/issues?q=is%3Aissue+). If not, please create a new [Github Issue](https://github.com/MeteoSwiss-MDA/dvas/issues). This is the best way for everyone to keep track of new problems and past solutions. 

## Contributing

We are currently in the early stages of development of dvas. If you would like to contribute to the code, please contact [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

Until its release, the dvas repository will remain private: branching will thus remain the only way to contribute to the code. To get a local copy of dvas and contribute to its improvement, one can collow the following steps:

0. Make sure you have git installed. Check that the setup is correct:
    
       git config --list
       
   If `user.name` and `user.email` are missing or do not match those your github account account, change them:

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

5. Modify the code locally. This could be the source code, or the docs `.rst` source files. 
   
   :warning: Please read carefully (and adhere to!) the dvas [style conventions](#styles) below.

6. Commit changes regularly, trying to bundle them in a meaningful manner.

       git add a_modified_file (OR possibly: git rm a_file_to_delete)
       git commit -m "Some useful, clear, and concise message. Use present tense."
   
   You can/should also push your branch to the dvas repository, if you want others to see what you are up to:
   
       git push origin your_branch_name

7. Once ready with the modifications, push your branch to the dvas repository. If warranted (it most likely will be!), remember to update the `CHANGELOG` before doing so. 

       git push origin your_branch_name

   Next, go to `your_branhc-name` on the dvas Github repository, and draft a new pull request. By default, the pull request should go from
   `your_branch_name` to the `develop` branch. Do not forget to link the pull request to a specific issue if warranted. Once the pull request is issued, automated checks will be run (pytest, 
   pylint, changelog, ...), which sould all succeed (if not, there might be something wrong with your changes).
   
   The code devs will then come and take a look at the pull request and assess its fitness for purpose.


## Styles

- **linting:** 
  * the following [pylint](https://www.pylint.org/) error codes are forbidden in dvas: ``E, C0303, C0304, C0112, C0114, C0115, C0116, C0411, W0611, W0612.`` Any pull request will be automatically linted, and these will be flagged accordingly. 
  * In general, we would encourage contributors to follow PEP8 as closely as possible/reasonable. You should check often
    how well you are doing using the command `pylint some_modified_file.py`.
  
- **doctrings:** Google Style. Please try to stick to the following MWE:
```
    ''' A brief one-liner description, that finishes with a dot.

    Use some
    multi-line space for
    more detailed info.

    Args:
        x (float, int): variable x could be of 2 types ...

           - *float*: x could be a float
           - *int*: x could also be an int

        y (list[str]; optional): variable y info

    Returns:
        bool: some grand Truth about he World.

    Example:
        If needed, you can specify chunks of code using code blocks::

            def some_function():
                print('hurray!')

    Note:
        `Source <https://github.com/sphinx-doc/sphinx/issues/3921>`__ 
        Please note the double _ _ after the link !

    Caution:
        Something the be careful about.

    '''
```
You should of course feel free to use more of the tools offered by [sphinx](https://www.sphinx-doc.org/en/master/), [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html), and 
[Google Doc Strings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google). But if you do, **please make sure there are no errors upon generating the docs !**

