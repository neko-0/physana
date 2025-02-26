# Deriving SF for control regions

1. generate histograms: iterative_sf/build_histogram.py
2. deriving SFs : iterative_sf/derive_correction.py

# Generate main histograms for unfolding.

1. generate histograms:
  1.1. first run: build_histogram.py
  1.2. in cmd: collinearw utility intersection --output run2.pkl --input "path_to/pkls"

2. generate migration histograms: build_histogram_migration.py
  2.1. first run: build_histogram.py
  2.2. in cmd: collinearw utility intersection --output run2.pkl --input "path_to/pkls"

3. update SFs:
  3.1 This step is to modified the SF (e.g. average EL & MU ttbar & zjets SFs)
  3.2 also prepare CR systematics which will be used later.

4. Generating CR & SFs systematics:
  4.1 run correction_factor_variation0.py, two new config files will be generated.

5. Merging all before unfolding:
  5.1 merge CR & SF systematics.
  5.2 merge phasespace migration bins.

6. Generate unfolding bias
  do_wjets_scaling_hidden.py

7. Compute unfolding systmeatic bands
  do_unfold_band.py

8. Merging unfolding bias
  derive_basic_unfolding_uncertainty.py

9. Averaging. (need external packages)
  well, you need to go through the installation, then do the serialization OUT
  for average, and serialization IN to convert back to our package format.

# Bootstrap

similar as above, but with only nominal samples.

merge bootstrap migration

do bootstrap unfolding.
