
import src.configs.config  
from loguru import logger

src_doc = "src/examples/retrieval/docs/src_golden_hymns_of_epictetus.txt"
output_doc = "src/examples/retrieval/docs/output_golden_hymns_of_epictetus_new.txt"

start_saving = False
stop_saving = False
line_to_save =[]

with open(src_doc, "r", encoding="utf-8") as f:
     for line in f.readlines():
         if "Are these the only works of Providence within us?" in line:
             start_saving = True
         if "*** END OF THE PROJECT GUTENBERG EBOOK THE GOLDEN SAYINGS OF EPICTETUS" in line:
             stop_saving = True
         if start_saving and not stop_saving:
             line_to_save.append(line)


# Write the lines to a new file
logger.info("len of line_to_save:" + str(len(line_to_save)))

with open(output_doc, "w", encoding="utf-8") as f:
    f.writelines(line_to_save)

wordcount = 0
with open(output_doc, "r", encoding="utf-8") as f:
    for line in f.readlines():
        wordcount += len(line.split())

logger.info(f"wordcount: {wordcount}")


logger.info("done")