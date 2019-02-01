import sys
import msgpack
import time
import os
from .util import timing
from web import util
from .state import Stats, Model, State, reload_all
from .files import FilesCache
from .choix_custom import setup

util.timing_enabled = False
def main(base, info_file, pref_file, output_file, completion_file,ready_file):
    #with open(af_file, "rb") as reader:
    #    af2 = msgpack.unpack(reader, use_list=False, raw=False)
    files = FilesCache(base)
    state = State(pref_file, files, None, False)
    errcount = 0
    while not os.path.exists(pref_file):
        time.sleep(0.3)
    with open(pref_file, "r") as reader_permanent:
        while True:
            try:
                #more_needed = True
                #while more_needed:
                #print("more_needed")
                #with timing("slow_calc::read"):
                #    with open(input_file, "rb") as reader:
                #        packed_stats, packed_model = msgpack.unpack(reader, use_list=False, raw=False)
                #with timing("slow_calc::make_objs"):
                #    stats = Stats(*packed_stats)
                #    model = Model(*packed_model, af2=af2)
                reload_all()
                state.read_from_file(reader_permanent)
                with open(info_file, "r") as reader:
                    info = set(list(x.strip() for x in reader))
                with timing("slow_calc::main"):
                    more_needed = state.model_.slow_calculations(state.stats, [], types=set(info))
                with timing("slow_calc::save"):
                    with open(output_file + "_", "wb") as writer:
                        packed_model =state.model_.to_msgpack("to_main") 
                        print("searching_pool_sim to_main to_msgpack", len(packed_model[9]) if packed_model is not None else None)
                        msgpack.pack(packed_model, writer, use_bin_type=True)
                    if os.path.exists(output_file):
                        os.unlink(output_file)
                    os.rename(output_file + "_", output_file)
                    with open(ready_file, "wb") as writer:
                        writer.write(b" ")
                with open(completion_file, "wb") as writer:
                    writer.write(b" ")
                while os.path.exists(completion_file):
                    # TODO: less dumb thing than this
                    time.sleep(0.05)
            except KeyboardInterrupt:
                return
            except:
                import traceback; traceback.print_exc()
                time.sleep(10)
                errcount += 1
                if errcount > 250:
                    return
                continue
            errcount = 0

if __name__ == "__main__":
    setup()
    main(*sys.argv[1:])
