import logger
import numpy as np


LOAD_MODEL = True


if LOAD_MODEL:
    logger.WRITE_MODE = 'a'
else:
    logger.WRITE_MODE = 'w'
logger.configure(dir_ = "./test/")

logger.set_level(logger.DEBUG)
logger.debug("should appear")





i = 0

def log():
    global i
    i += 1
    val = np.random.rand()
    logger.logkv("a", val)
    logger.info(f'value: {val}')
    logger.logkv("i", i)
    logger.info(f'i: {i}')
    logger.dumpkvs()
    
logger.logkv("a", np.random.rand())
logger.dumpkvs()

logger.logkv("a", np.random.rand())
logger.dumpkvs()
for _ in range(10):
    log()
    
logger.logkv("c", np.random.rand())
logger.dumpkvs()

logger.logkv("c", np.random.rand())
logger.dumpkvs()

logger.logkv("d", np.random.rand())
logger.dumpkvs()