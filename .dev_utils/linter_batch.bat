pylint --errors-only ./src/dvas
pylint --disable=all --enable=C0303,C0304,C0112,C0114,C0115,C0116,C0411,W0611,W0612 ./src/dvas
pylint --errors-only ./src/dvas_recipes
pylint --disable=all --enable=C0303,C0304,C0112,C0114,C0115,C0116,C0411,W0611,W0612 ./src/dvas_recipes

pylint --errors-only ./test
pylint --disable=all --enable=C0303,C0304,C0112,C0114,C0115,C0116,C0411,W0611,W0612 ./test
