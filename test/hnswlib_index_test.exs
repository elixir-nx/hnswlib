defmodule HNSWLib.Index.Test do
  use ExUnit.Case
  doctest HNSWLib.Index

  test "HNSWLib.Index.new/3 with L2-space" do
    space = :l2
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :cosine
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :ip
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/3 with cosine-space" do
    space = :cosine
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/3 with inner-product space" do
    space = :ip
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/3 with non-default keyword parameters" do
    space = :ip
    dim = 8
    max_elements = 200

    m = 200
    ef_construction = 400
    random_seed = 42
    allow_replace_deleted = true

    {:ok, index} =
      HNSWLib.Index.new(space, dim, max_elements,
        m: m,
        ef_construction: ef_construction,
        random_seed: random_seed,
        allow_replace_deleted: allow_replace_deleted
      )

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12

    {:ok, index} =
      HNSWLib.Index.new(space, dim, max_elements,
        m: m,
        ef_construction: ef_construction,
        random_seed: random_seed,
        allow_replace_deleted: allow_replace_deleted
      )

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/3 with invalid keyword parameter m" do
    space = :ip
    dim = 8
    max_elements = 200

    m = -1

    assert {:error,
            "expect keyword parameter `:m` to be a non-negative integer, got `#{inspect(m)}`"} ==
             HNSWLib.Index.new(space, dim, max_elements, m: m)
  end

  test "HNSWLib.Index.new/3 with invalid keyword parameter ef_construction" do
    space = :ip
    dim = 8
    max_elements = 200

    ef_construction = -1

    assert {:error,
            "expect keyword parameter `:ef_construction` to be a non-negative integer, got `#{inspect(ef_construction)}`"} ==
             HNSWLib.Index.new(space, dim, max_elements, ef_construction: ef_construction)
  end

  test "HNSWLib.Index.new/3 with invalid keyword parameter random_seed" do
    space = :ip
    dim = 8
    max_elements = 200

    random_seed = -1

    assert {:error,
            "expect keyword parameter `:random_seed` to be a non-negative integer, got `#{inspect(random_seed)}`"} ==
             HNSWLib.Index.new(space, dim, max_elements, random_seed: random_seed)
  end

  test "HNSWLib.Index.new/3 with invalid keyword parameter allow_replace_deleted" do
    space = :ip
    dim = 8
    max_elements = 200

    allow_replace_deleted = -1

    assert {:error,
            "expect keyword parameter `:allow_replace_deleted` to be a boolean, got `#{inspect(allow_replace_deleted)}`"} ==
             HNSWLib.Index.new(space, dim, max_elements,
               allow_replace_deleted: allow_replace_deleted
             )
  end

  test "HNSWLib.Index.knn_query/2 with binary" do
    space = :ip
    dim = 2
    max_elements = 200
    data = <<42.0::float-32, 42.0::float-32>>
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert :ok == HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with [binary]" do
    space = :ip
    dim = 2
    max_elements = 200
    data = [<<42.0::float-32, 42.0::float-32>>, <<42.0::float-32, 42.0::float-32>>]
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert :ok == HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with Nx.Tensor (:f32)" do
    space = :ip
    dim = 2
    max_elements = 200
    data = Nx.tensor([1, 2], type: :f32)
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert :ok == HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with Nx.Tensor (:u8)" do
    space = :ip
    dim = 2
    max_elements = 200
    data = Nx.tensor([1, 2], type: :u8)
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert :ok == HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with invalid length of data" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = <<42::16, 1::24>>

    assert {:error, "vector feature size should be a multiple of 4 (sizeof(float))"} ==
             HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with invalid dimensions of data" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = <<42::float-32, 42::float-32, 42::float-32>>

    assert {:error, "Wrong dimensionality of the vectors, expect `2`, got `3`"} ==
             HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with inconsistent dimensions of [data]" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = [<<42::float-32, 42::float-32>>, <<42::float-32, 42::float-32, 42::float-32>>]

    assert {:error, "all vectors in the input list should have the same size"} ==
             HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with invalid dimensions of [data]" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = [<<42::float-32, 42::float-32, 42::float-32>>, <<42::float-32, 42::float-32, 42::float-32>>]

    assert {:error, "Wrong dimensionality of the vectors, expect `2`, got `3`"} ==
             HNSWLib.Index.knn_query(index, data)
  end

  test "HNSWLib.Index.knn_query/2 with invalid type for `k`" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = <<42.0, 42.0>>
    k = :invalid

    assert {:error, "expect keyword parameter `:k` to be a positive integer, got `:invalid`"} ==
             HNSWLib.Index.knn_query(index, data, k: k)
  end

  test "HNSWLib.Index.knn_query/2 with invalid type for `num_threads`" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = <<42.0, 42.0>>
    num_threads = :invalid

    assert {:error, "expect keyword parameter `:num_threads` to be an integer, got `:invalid`"} ==
             HNSWLib.Index.knn_query(index, data, num_threads: num_threads)
  end

  test "HNSWLib.Index.knn_query/2 with invalid type for `filter`" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)
    data = <<42.0, 42.0>>
    filter = :invalid

    assert {:error,
            "expect keyword parameter `:filter` to be a function that can be applied with 1 number of arguments , got `:invalid`"} ==
             HNSWLib.Index.knn_query(index, data, filter: filter)
  end

  test "HNSWLib.Index.get_ids_list/1 when empty" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.Index.new(space, dim, max_elements)

    assert {:ok, []} == HNSWLib.Index.get_ids_list(index)
  end
end
