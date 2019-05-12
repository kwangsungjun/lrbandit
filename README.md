# lrbandit

This is the code for the paper 'Bilinear Bandits with Low-rank Structure' published in ICML 2019.

Required software
 - python 3 with numpy, scipy, sklearn, cython

Compile needed
 - in `matrixrecovery`, run `cython myutils_cython.pyx` and then `python3 setup.py install`
 - for mac
    - in `pyOptSpace_py3_custom`, run `cython optspace.pyx` then `python3 setup.py install`
    - copy the created `*.so` file to the upper directory
 - for linux
    - do the same as mac above, but in the directory `pyOptSpace_py3_linux_custom`

To replicate the plot in the paper (expreiment data already run)
 - run `python3 analyze_expr01_20190119-9_paper.py`

To replicate the result in the paper
  1. run the script (recommended to use nTry=2 to ensure there is no runtime error)
```
for A in  bltwostage-bm-sp_simple2 bltwostage-sp_simple2 blonestage-sp_simple2 bloful 
do
python3  run_expr01.py ${A} -- -d 8 -R 0.01 -r 1 -T 10000 -tg 1 --nTry=60 |& tee log_0119_${A}.txt;
done
```
  2. move all .pkl files to a subfolder with name `XXX` (with your choice)
  3. modify `analyze_expr01_20190119-9_paper.py` line 4-10 so that the prefix points to the 
     folder `./XXX/` and the file names are appropriately changed.


<!--
# License

This SDK is distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0), see [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for more information.
-->
