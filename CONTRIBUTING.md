# Contributing to dvas

## Code of conduct

This project and everyone participating in it is governed by the [dvas Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

## Reporting a bug

If you find something odd with dvas, first check it is a [known issue](https://github.com/MeteoSwiss-MDA/dvas/issues?q=is%3Aissue+). if not, please create a new [Github Issue](https://github.com/MeteoSwiss-MDA/dvas/issues). This is the best way for everyone to keep track of new problems and past solutions. 

## Contributing

We are currently in the very early stages of development of dvas. If you would like to contribute to the code, please contact [frederic.vogt@meteoswiss.ch](mailto:frederic.vogt@meteoswiss.ch).

## Styles

- **linting:** 
  * the following [pylint](https://www.pylint.org/) error codes are forbidden in dvas: ``E, C0303, C0304, C0112, C0114, C0115, C0116, C0411, W0611, W0612.`` Any pull request will be automatically linted, and these will be flagged accordingly. 
  * In general, we would encourage contributors to follow PEP8 as closely as possible/reasonable.
  
- **doctrings:** Google Style. Please try to stick to the following MWE:
```
    ''' A brief one-liner description, that finishes with a dot.

    Use some
    multi-line space for
    more detailed info.

    Args:
        x (float, int): variable x

           - *float*: x could be an ``float``
           - *int*: x could also be an ``int``

        y (list[str]): variable y

    Returns:
        bool: some grand Truth aboutt he World.

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
