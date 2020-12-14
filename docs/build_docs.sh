# Custom bash script to generate the dvas docs
# Created June 2020; F.P.A. Vogt; frederic.vogt@meteoswiss.ch
#

# Step 0: clean the existing apidoc rst files and any previous build folder.
# Make sure to not delete any important .git info for the CI auto_publish action !
#rm -rf ./source/modules
#rm -rf ./build/*.html
#rm -rf ./build/_sources
#rm -rf ./build/_images
#rm -rf ./build/_static
#rm -rf ./build/.buildinfo
#rm -rf ./build/.doctrees
cd ./build
find . -not \( -path '*.git*' -o -name '.nojekyll' -o -name '.-gitignore' \) -delete
cd ..

# Step 1: run autodoc to generate all the docstring rst files.
# Force the rewrite of all of them to capture *all* the recent changes.
sphinx-apidoc -f -M -o ./source/modules/ ../src/

# Delete the superfluous module.rst file that the previous command creates.
rm -f ./source/modules/modules.rst

# Run the dvas help, to automate the display of the response in the docs.
dvas -h > source/dvas_help_msg.txt

# Generate the documentation, storing it in the build directory
sphinx-build -a -b html ./source ./build

# Clean-up
rm source/dvas_help_msg.txt
