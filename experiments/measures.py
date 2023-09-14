
    segmentations = {}
    for subject in range(infer_loaders["infer"].dataset.subjects):
        segmentations[subject] = torch.zeros(
            tuple(np.insert(volume_shape, 0, args.n_classes)), dtype=torch.uint8,
        )

    segmentations = voxel_majority_predict_from_subvolumes(
        infer_loaders["infer"], args.n_classes, segmentations
    )
    subject_metrics = []
    for subject, subject_data in enumerate(tqdm(infer_loaders["infer"].dataset.data)):
        seg_labels = nib.load(subject_data["nii_labels"]).get_fdata()
        segmentation_labels = torch.nn.functional.one_hot(
            torch.from_numpy(seg_labels).to(torch.int64), args.n_classes
        )

        inference_dice = (
            dice(
                torch.nn.functional.one_hot(segmentations[subject], args.n_classes).permute(
                    0, 3, 1, 2
                ),
                segmentation_labels.permute(0, 3, 1, 2),
            )
            .detach()
            .numpy()
        )
        macro_inference_dice = (
            dice(
                torch.nn.functional.one_hot(segmentations[subject], args.n_classes).permute(
                    0, 3, 1, 2
                ),
                segmentation_labels.permute(0, 3, 1, 2),
                mode="macro",
            )
            .detach()
            .numpy()
        )
        subject_metrics.append((inference_dice, macro_inference_dice))

    per_class_df = pd.DataFrame([metric[0] for metric in subject_metrics])
    macro_df = pd.DataFrame([metric[1] for metric in subject_metrics])
    print(per_class_df, macro_df)
