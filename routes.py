from flask import Blueprint
from controllers.plants import lession_cnn


deep_router=Blueprint("deep",__name__)

#định nghĩa router
deep_router.route("/lession-ann",methods=["GET","POST"])(lession_cnn)
