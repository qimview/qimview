import glob
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str, help='directory to look for suffixes')


    verbose = False
    args = parser.parse_args()
    _params = vars(args)

    im_list = glob.glob(f"{_params['dir']}/**/*.jpg", recursive=True)
    # print(im_list)
    bn_list = [ os.path.splitext(os.path.basename(n))[0] for n in im_list]

    def list_suffixes(name):
        res = []
        splitted = name.split('_')
        for n in range(1,len(splitted)):
            res.append('_'.join(splitted[n:]))
        return res

    suffixes = []
    for bn in bn_list:
        suffixes.extend(list_suffixes(bn))
    suffixes = list(set(suffixes))

    im_list_scores = {}
    for suf in suffixes:
        if verbose: print(f" ----- suffix {suf}")
        im_list =  list(filter(lambda s: s.endswith(suf), bn_list))
        if verbose: print(f"im_list {im_list}")
        new_bn_list = [ f[:-len(suf)] for f in im_list]
        list_str = ','.join(new_bn_list)
        # print(f"list_str {list_str}")
        if list_str not in im_list_scores:
            im_list_scores[list_str] = [suf]
        else:
            im_list_scores[list_str].append(suf)

    for k in im_list_scores:
        if verbose: print(f"{k}:  {im_list_scores[k]}")
    candidates = sorted(im_list_scores, key=lambda x: x.count(',')*len(im_list_scores[x]), reverse=True)
    for n in range(len(candidates)):
        if n<30:
            print("----------")
            c = candidates[n]
            print(','.join(im_list_scores[c]))
            print(f"{c.count(',')*len(im_list_scores[c])}")

