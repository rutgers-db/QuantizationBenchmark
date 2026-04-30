python run.py --dataset laion-768-ip --algorithm TurboQuant --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm OptimizedProductQuantizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm OptimizedScalarQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm ProductQuantizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm ProductQuantizationFastScanFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm QdrantBinaryQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm RabitQLibrary --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm SAQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm ScalarQuatizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm WeaviateRotationalQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug

# python run.py --dataset text2image-200-euclidean --algorithm TurboQuant --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm OptimizedProductQuantizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm OptimizedScalarQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm ProductQuantizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm ProductQuantizationFastScanFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm QdrantBinaryQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm RabitQLibrary --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm SAQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm ScalarQuatizationFaiss --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm WeaviateRotationalQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug

python run.py --dataset sift-128-euclidean --algorithm WeaviateRotationalQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset sift-128-euclidean --algorithm QdrantBinaryQuantization --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug


python run.py --algorithm Faiss-IVFPQ --build-images --force-rebuild
python run.py --algorithm Faiss-IVFSQ --build-images --force-rebuild
python run.py --algorithm Faiss-OPQ-IVFPQ --build-images --force-rebuild
python run.py --algorithm IVFOSQ --build-images --force-rebuild
python run.py --algorithm IVFRabitQLibrary --build-images --force-rebuild
# python run.py --algorithm IVFSAQ --build-images --force-rebuild
python run.py --algorithm IVFTurboQuant --build-images --force-rebuild
python run.py --algorithm Faiss-IVFPQFastScan --build-images --force-rebuild
python run.py --algorithm IVFQdrantBQ --build-images --force-rebuild
python run.py --algorithm IVFWeaviateRSQ --build-images --force-rebuild

python run.py --dataset laion-768-ip --algorithm Faiss-IVFPQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm Faiss-IVFSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm Faiss-OPQ-IVFPQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm IVFOSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm IVFRabitQLibrary --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset laion-768-ip --algorithm IVFSAQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm IVFTurboQuant --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm Faiss-IVFPQFastScan --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm IVFQdrantBQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset laion-768-ip --algorithm IVFWeaviateRSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug

python run.py --dataset text2image-200-euclidean --algorithm Faiss-IVFPQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm Faiss-IVFSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm Faiss-OPQ-IVFPQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm IVFOSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm IVFRabitQLibrary --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
# python run.py --dataset text2image-200-euclidean --algorithm IVFSAQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm IVFTurboQuant --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm Faiss-IVFPQFastScan --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm IVFQdrantBQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
python run.py --dataset text2image-200-euclidean --algorithm IVFWeaviateRSQ --data-dir /data/local/embedding_dataset/hdf5/ --distribution-shift-test --debug
