from flask import Flask, request
import numpy as np

from opts import options

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


@app.route("/", methods=["GET"])
def index():

    # make sure the request is valid
    def is_valid():
        if request.method == "GET":
            return True
        return False

    if not is_valid():
        return  # bad request status code and error message

    # parse arguments

    # perform analysis of choice

    # return results

    return


if __name__ == '__main__':
    app.run(
        debug=True,
    )
