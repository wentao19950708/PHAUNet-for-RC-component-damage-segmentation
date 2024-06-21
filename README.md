# PHAUNet-for-RC-component-damage-segmentation

**Paper:** PHA-UNet: Pyramid Haar wavelet downsampling attention UNet for damage segmentation of post-earthquake damaged RC components

**Abstract:** Damage identification in post-earthquake reinforced concrete (RC) structures based on semantic segmentation has been recognized as a promising approach for rapid and non-contact damage localization and quantification. In damage segmentation tasks, damage regions are often set against complex backgrounds, featuring irregular geometric boundaries and intricate textures, posing significant challenges to model segmentation performance. Additionally, the absence of public datasets exacerbates these challenges, hindering advancements in this field. In this study, a pyramid Haar wavelet downsampling attention UNet (PHA-UNet) semantic segmentation network is proposed, and a database containing 1400 images of damaged RC components (PEDRC-Dataset) with pixel-level annotations is established. In the proposed PHA-UNet, attention mechanisms, multiscale feature fusion, Haar wavelet downsampling, and transfer learning are introduced to address above challenges. Finally, the proposed PHA-UNet is compared with 4 existing image segmentation architectures on both the Cityspace and the PEDRC-Dataset.

**dataset**：dataset is open access on https://www.researchgate.net/lab/Lei-Li-Lab-3

**attention**：It is not recommended to train directly using the dataset. Pretraining is advised, and it is recommended to use the Cityspace dataset. 

If you use images from the dataset as illustrations, please contact wangwentao@xauat.edu.cn for confirmation, as some images in the dataset are temporarily not allowed to be used as paper illustrations.
