defmodule HNSWLib.BFIndex.Test do
  use ExUnit.Case
  doctest HNSWLib.BFIndex

  test "HNSWLib.BFIndex.new/3 with L2-space" do
    space = :l2
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :cosine
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    space = :ip
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.BFIndex.new/3 with cosine-space" do
    space = :cosine
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.BFIndex.new/3 with inner-product space" do
    space = :ip
    dim = 8
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert is_pid(index.pid)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.BFIndex.knn_query/2 with binary" do
    space = :l2
    dim = 2
    max_elements = 200

    data =
      Nx.tensor(
        [
          [42, 42],
          [43, 43],
          [0, 0],
          [200, 200],
          [200, 220]
        ],
        type: :f32
      )

    ids = [5, 6, 7, 8, 9]

    query = <<41.0::float-32-little, 41.0::float-32-little>>
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    assert :ok == HNSWLib.BFIndex.add_items(index, data, ids: ids)

    {:ok, labels, dists} = HNSWLib.BFIndex.knn_query(index, query, k: 3)
    assert 1 == Nx.to_number(Nx.all_close(labels, Nx.tensor([5, 6, 7])))
    assert 1 == Nx.to_number(Nx.all_close(dists, Nx.tensor([2.0, 8.0, 3362.0])))
  end

  test "HNSWLib.BFIndex.knn_query/2 with [binary]" do
    space = :l2
    dim = 2
    max_elements = 200

    data =
      Nx.tensor(
        [
          [42, 42],
          [43, 43],
          [0, 0],
          [200, 200],
          [200, 220]
        ],
        type: :f32
      )

    ids = [5, 6, 7, 8, 9]

    query = [
      <<0.0::float-32-little, 0.0::float-32-little>>,
      <<41.0::float-32-little, 41.0::float-32-little>>
    ]

    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    assert :ok == HNSWLib.BFIndex.add_items(index, data, ids: ids)

    {:ok, labels, dists} = HNSWLib.BFIndex.knn_query(index, query, k: 3)
    assert 1 == Nx.to_number(Nx.all_close(labels, Nx.tensor([[7, 5, 6], [5, 6, 7]])))

    assert 1 ==
             Nx.to_number(
               Nx.all_close(dists, Nx.tensor([[0.0, 3528.0, 3698.0], [2.0, 8.0, 3362.0]]))
             )
  end

  test "HNSWLib.BFIndex.knn_query/2 with Nx.Tensor (:f32)" do
    space = :l2
    dim = 2
    max_elements = 200

    data =
      Nx.tensor(
        [
          [42, 42],
          [43, 43],
          [0, 0],
          [200, 200],
          [200, 220]
        ],
        type: :f32
      )

    query = Nx.tensor([1, 2], type: :f32)
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    assert :ok == HNSWLib.BFIndex.add_items(index, data)

    {:ok, labels, dists} = HNSWLib.BFIndex.knn_query(index, query)
    assert 1 == Nx.to_number(Nx.all_close(labels, Nx.tensor([2])))
    assert 1 == Nx.to_number(Nx.all_close(dists, Nx.tensor([5])))
  end

  test "HNSWLib.BFIndex.knn_query/2 with Nx.Tensor (:u8)" do
    space = :l2
    dim = 2
    max_elements = 200

    data =
      Nx.tensor(
        [
          [42, 42],
          [43, 43],
          [0, 0],
          [200, 200],
          [200, 220]
        ],
        type: :f32
      )

    query = Nx.tensor([1, 2], type: :u8)
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    assert :ok == HNSWLib.BFIndex.add_items(index, data)

    {:ok, labels, dists} = HNSWLib.BFIndex.knn_query(index, query)
    assert 1 == Nx.to_number(Nx.all_close(labels, Nx.tensor([2])))
    assert 1 == Nx.to_number(Nx.all_close(dists, Nx.tensor([5])))
  end

  test "HNSWLib.BFIndex.knn_query/2 with invalid length of data" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    data = <<42::16, 1::24>>

    assert {:error, "vector feature size should be a multiple of 4 (sizeof(float))"} ==
             HNSWLib.BFIndex.knn_query(index, data)
  end

  test "HNSWLib.BFIndex.knn_query/2 with invalid dimensions of data" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    data = <<42::float-32, 42::float-32, 42::float-32>>

    assert {:error, "Wrong dimensionality of the vectors, expect `2`, got `3`"} ==
             HNSWLib.BFIndex.knn_query(index, data)
  end

  test "HNSWLib.BFIndex.knn_query/2 with inconsistent dimensions of [data]" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    data = [<<42::float-32, 42::float-32>>, <<42::float-32, 42::float-32, 42::float-32>>]

    assert {:error, "all vectors in the input list should have the same size"} ==
             HNSWLib.BFIndex.knn_query(index, data)
  end

  test "HNSWLib.BFIndex.knn_query/2 with invalid dimensions of [data]" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    data = [
      <<42::float-32, 42::float-32, 42::float-32>>,
      <<42::float-32, 42::float-32, 42::float-32>>
    ]

    assert {:error, "Wrong dimensionality of the vectors, expect `2`, got `3`"} ==
             HNSWLib.BFIndex.knn_query(index, data)
  end

  test "HNSWLib.BFIndex.knn_query/2 with invalid type for `k`" do
    space = :ip
    dim = 2
    max_elements = 200
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    data = <<42.0, 42.0>>
    k = :invalid

    assert_raise ArgumentError, "expect keyword parameter `:k` to be a positive integer, got `:invalid`", fn ->
      HNSWLib.BFIndex.knn_query(index, data, k: k)
    end
  end

  test "HNSWLib.BFIndex.add_items/3 without specifying ids" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert :ok == HNSWLib.BFIndex.add_items(index, items)
  end

  test "HNSWLib.BFIndex.add_items/3 with specifying ids (Nx.Tensor)" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = Nx.tensor([100, 200])
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert :ok == HNSWLib.BFIndex.add_items(index, items, ids: ids)
  end

  test "HNSWLib.BFIndex.add_items/3 with specifying ids (list)" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = [100, 200]
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert :ok == HNSWLib.BFIndex.add_items(index, items, ids: ids)
  end

  test "HNSWLib.BFIndex.add_items/3 with wrong dim of data tensor" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20, 300], [30, 40, 500]], type: :f32)
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert {:error, "Wrong dimensionality of the vectors, expect `2`, got `3`"} ==
             HNSWLib.BFIndex.add_items(index, items)
  end

  test "HNSWLib.BFIndex.add_items/3 with wrong dim of ids" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = Nx.tensor([[100], [200]])
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert {:error, "expect ids to be a 1D array, got `{2, 1}`"} ==
             HNSWLib.BFIndex.add_items(index, items, ids: ids)
  end

  test "HNSWLib.BFIndex.delete_vector/2" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    query = Nx.tensor([29, 39], type: :f32)
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)

    assert :ok == HNSWLib.BFIndex.add_items(index, items)
    assert :ok == HNSWLib.BFIndex.delete_vector(index, 0)

    {:ok, labels, dists} = HNSWLib.BFIndex.knn_query(index, query)
    assert 1 == Nx.to_number(Nx.all_close(labels, Nx.tensor([1])))
    assert 1 == Nx.to_number(Nx.all_close(dists, Nx.tensor([2])))
  end

  test "HNSWLib.BFIndex.save_index/2" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = Nx.tensor([100, 200])
    save_to = Path.join([__DIR__, "saved_bfindex.bin"])
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    :ok = HNSWLib.BFIndex.add_items(index, items, ids: ids)

    # ensure file does not exist
    File.rm(save_to)
    assert :ok == HNSWLib.BFIndex.save_index(index, save_to)
    assert File.exists?(save_to)

    # cleanup
    File.rm(save_to)
  end

  test "HNSWLib.BFIndex.load_index/3" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = Nx.tensor([100, 200])
    save_to = Path.join([__DIR__, "saved_bfindex.bin"])
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    :ok = HNSWLib.BFIndex.add_items(index, items, ids: ids)

    # ensure file does not exist
    File.rm(save_to)
    assert :ok == HNSWLib.BFIndex.save_index(index, save_to)
    assert File.exists?(save_to)

    {:ok, index_from_save} = HNSWLib.BFIndex.new(space, dim, max_elements)
    assert :ok == HNSWLib.BFIndex.load_index(index_from_save, save_to)

    # cleanup
    File.rm(save_to)
  end

  test "HNSWLib.BFIndex.load_index/3 with new max_elements" do
    space = :l2
    dim = 2
    max_elements = 200
    items = Nx.tensor([[10, 20], [30, 40]], type: :f32)
    ids = Nx.tensor([100, 200])
    save_to = Path.join([__DIR__, "saved_bfindex.bin"])
    {:ok, index} = HNSWLib.BFIndex.new(space, dim, max_elements)
    :ok = HNSWLib.BFIndex.add_items(index, items, ids: ids)

    # ensure file does not exist
    File.rm(save_to)
    assert :ok == HNSWLib.BFIndex.save_index(index, save_to)
    assert File.exists?(save_to)

    {:ok, index_from_save} = HNSWLib.BFIndex.new(space, dim, max_elements)

    new_max_elements = 1

    assert :ok ==
             HNSWLib.BFIndex.load_index(index_from_save, save_to, max_elements: new_max_elements)

    # cleanup
    File.rm(save_to)
  end
end
