# Load signature analysis for residential loads

Non-intrusive Appliance Load Monitoring (NILM) aimed to monitor the main supply circuit included various appliances (devices) which switch on and off independently. The voltage and current waveforms of the total demand side was captured and analyzed to determine the number and characteristic of each operating appliances without accessing each component by installing sensors. This can offer an effective and convenient way to gather load data compared with previous way. The result of the data record is very useful to utilities, manufacturers, policy makers and consumers by all means. Load signature (LS) is a unique identifier of individual appliances to be distinguished in a composite load signal. This area of research aimed to identify the appliances via the main circuit. Load Disaggregation requires pattern recognition algorithm. This topic has been explored though the last decade. Plenty of approaches have been proposed and quite a number of them can achieve trustworthy performance.

## Approach

Most of the appliances operate with stable power (with trivial variation). It is worthwhile to separate the appliances into two groups and the state of an appliance is ‘On’ or ‘Off’ state only. Hence one group, named Group A, consists of appliances with stable power state and another, named Group B, consists of appliances with oscillating power state.

During the preprocessing, this model requires extracting the stable signal and oscillating signal from the composite signal. Low pass filter is applied on the composite signal with length of window. The filter collects the power levels within the window and takes the minimum point as result. The resultant signal thus becomes the stable power signal. Hence, the oscillating power signal can be retrieve by subtracting the stable signal from the composite signal. This approach has an assumption that, within the window, there is one state change in Group A appliances.

![general](https://user-images.githubusercontent.com/44134941/46916787-eb817d00-cff1-11e8-81d3-e51196d56dcf.png)

For Group A appliances, the load disaggregation requires event detection to compute the features and activate the load signature classification with classifier A. SVM and Naïve Bayes algorithms are applied in the classifier for comparison. 
Based on the previous researches, both CO and FHMM have decline performance with increasing number of appliances. Consequently, with this proposed approach and small number of Group B appliance, CO and FHMM can be employed to disaggregate the oscillating signal with expectantly reasonable performance. CO and FHMM generate predicted individual power signals pattern for each appliance. This results in two power levels as features at any time t for Classifier B.  Classifier B aims to improve the performance from the disaggregated states. Naïve Bayes and Neural Network algorithm are applied for comparison as well.

## Demo

DemoA identifies load signature performance of Group A appliances based on various classifiers with event detection.
DemoB identifies load signature performance of Group B appliances based on various classifiers with pattern recognization.

## Future works

Classifier A, which aimed to identify Group A appliance states, uses active power and reactive power in stable power signal as features to classify with SVM and Naïve Bayes. Classifier B, which aimed to identify Group B appliance state, utilizes disaggregated signals from FHMM and CO as features to classify with Naïve Bayes and Neural Network. 

The disaggregation algorithms for group B appliance may possibly be further modified. Because there are numbers of enhanced algorithms based on FHMM, it is possible to replace the existing one with the latest. Also, the features inputs to the classifier can be modular and combine more disaggregation results. Finally, thanks to the blooming technology development, the tools become more efficient and easy to access to construct this hybrid method for load disaggregation and classification with load signatures. Hopefully, futures researches complement this method with strengthen performance.
