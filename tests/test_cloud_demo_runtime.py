from src.runtime import build_cloud_demo_runtime


def test_cloud_demo_runtime_has_recommendation_components():
    runtime = build_cloud_demo_runtime()

    assert not runtime["movies_df"].empty
    assert runtime["content_model"].is_fitted
    assert runtime["pop_model"].is_fitted
    assert runtime["svd_model"].is_fitted
    assert runtime["hybrid_model"].is_fitted
    assert runtime["genre_prior"].is_fitted
