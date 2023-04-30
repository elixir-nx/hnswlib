defmodule HNSWLib.Index.Test do
  use ExUnit.Case
  doctest HNSWLib.Index

  test "HNSWLib.Index.new/2 with L2-space" do
    space = :l2
    dim = 8
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim

    space = :cosine
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim

    space = :ip
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/2 with cosine-space" do
    space = :cosine
    dim = 8
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.new/2 with inner-product space" do
    space = :ip
    dim = 8
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim

    dim = 12
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert is_reference(index.reference)
    assert space == index.space
    assert dim == index.dim
  end

  test "HNSWLib.Index.init/3" do
    space = :l2
    dim = 8
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert :ok == HNSWLib.Index.init_index(index, 200)
  end

  test "HNSWLib.Index.init/3 should only be called once" do
    space = :l2
    dim = 8
    {:ok, index} = HNSWLib.Index.new(space, dim)

    assert :ok == HNSWLib.Index.init_index(index, 200)
    assert {:error, "The index is already initiated."} == HNSWLib.Index.init_index(index, 200)
    assert {:error, "The index is already initiated."} == HNSWLib.Index.init_index(index, 200)
  end
end
