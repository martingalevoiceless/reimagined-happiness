import sys
import msgpack
import time
import os
from .util import timing
from .state import Stats, Model

def main(input_file, af_file, output_file, completion_file):
    while True:
        with timing("slow_calc::read"):
            with open(input_file, "rb") as reader:
                packed_stats, packed_model = msgpack.unpack(reader, use_list=False, raw=False)
            with open(af_file, "rb") as reader:
                af2 = msgpack.unpack(reader, use_list=False, raw=False)
        with timing("slow_calc::main"):
            stats = Stats(*packed_stats)
            model = Model(*packed_model, af2=af2)
            model.slow_calculations(stats, [])
        with timing("slow_calc::save"):
            with open(output_file + "_", "wb") as writer:
                msgpack.pack(model.to_msgpack(), writer, use_bin_type=True)
            os.rename(output_file + "_", output_file)
            with open(completion_file, "wb") as writer:
                writer.write(b" ")
        while os.path.exists(completion_file):
            # TODO: less dumb thing than this
            time.sleep(0.05)

if __name__ == "__main__":
    main(*sys.argv[1:])
