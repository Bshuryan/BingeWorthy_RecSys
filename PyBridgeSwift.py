import logging
import sys
import objc
from Foundation import NSObject
import Content_Based_Rec_Sys


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Loaded python bundle")

# Load the protocol from Objective-C
BridgeInterface = objc.protocolNamed("BingeWorthy.BridgeInterface")


class Bridge(NSObject, protocols=[BridgeInterface]):
    @classmethod
    def createInstance(self):
        return Bridge.alloc().init()

    @staticmethod
    def getMovies(movie):
        return Content_Based_Rec_Sys.get_movie_recs(movie)