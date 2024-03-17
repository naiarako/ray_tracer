# ray_tracer

Ray tracing for pulse-echo ultrasound tomography. 

The file <code>ray_tracers.py</code> defines two types of ray tracers depending on the emitted wavefiled (i.e., plane waves or diverging waves from single transucer elements). For each emission type, we use two different parameterizations to solve the related line integrals. The name 'pixel' is used for a parameterization that assumes constant speed of sound in each pixel, and the name 'bilint' refers to the use of bilinear interpolation (see the documentation pdf file for more details). The jupyter-notebook <code>Example_ForwardOperator.ipynb</code> shows how to use these ray tracers to build the forward operator corresponding to the tomographic problem. Then, the jupyter-notebook <code>Inversion.ipynb</code> shows how to solve the related inverse problem using Tikhonov regularization. Specific codes for inversion are defined in <code>inversion.ipynb</code>.

Repository under construction.
