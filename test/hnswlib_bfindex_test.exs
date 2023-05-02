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
