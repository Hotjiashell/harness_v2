from typing import List, Dict, Any, Optional

async def cluster(
        texts: List[str], #待聚类的文本列表
        cluster_method: str = "kmeans", #支持kmeans和hdbscan
        n_clusters: Optional[int] = 5, #仅kmeans需要
        min_cluster_size: Optional[int] = 3, #仅hdbscan需要
        embedding_concurrency: int = 5, #并发获取文本向量的数量
        embedding_model: str = "text-embedding-3-small",
        embedding_url: Optional[str] = "http://localhost:8000/embeddings",
        embedding_api_key: Optional[str] = "1234",
) -> List[Dict[str, Any]]:
    """
    对输入的文本列表进行聚类，返回每个文本所属的聚类标签和对应的文本内容。
    Args:
        texts: 待聚类的文本列表
        cluster_method: 聚类方法，支持kmeans和hdbscan，默认为kmeans
        n_clusters: kmeans算法的聚类数量，仅cluster_method为kmeans时需要，默认为5
        min_cluster_size: hdbscan算法的最小聚类大小，仅cluster_method为hdbscan时需要，默认为3
        embedding_concurrency: 并发获取文本向量的数量，默认为5
        embedding_model: 获取文本向量使用的模型名称，默认为"text-embedding-3-small"
        embedding_url: 获取文本向量的API地址，默认为"http://localhost:8000/embeddings"
        embedding_api_key: 获取文本向量的API密钥，默认为"1234"
    Returns:
        包含每个文本所属聚类标签和文本内容的列表，格式如下：
        [
            {
                "cluster_id": 0, #聚类标签，使用kmean时，值介于0和n_clusters-1之间；使用hdbscan时，值为-1表示噪声点，非负整数表示聚类标签
                "text": "文本内容"
            },
            ...
        ]
    """
    pass