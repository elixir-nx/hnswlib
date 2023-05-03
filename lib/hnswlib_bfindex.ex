defmodule HNSWLib.BFIndex do
  @moduledoc """
  Documentation for `HNSWLib.BFIndex`.
  """

  defstruct [:space, :dim, :pid]
  alias __MODULE__, as: T
  alias HNSWLib.Helper

  use GenServer

  @doc """
  Construct a new BFIndex

  ##### Positional Parameters

  - *space*: `:cosine` | `:ip` | `:l2`.

    An atom that indicates the vector space. Valid values are

      - `:cosine`, cosine space
      - `:ip`, inner product space
      - `:l2`, L2 space

  - *dim*: `non_neg_integer()`.

    Number of dimensions for each vector.

  - *max_elements*: `non_neg_integer()`.

    Number of maximum elements.
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer(), non_neg_integer()) ::
          {:ok, %T{}} | {:error, String.t()}
  def new(space, dim, max_elements)
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_integer(max_elements) and max_elements >= 0 do
    with {:ok, pid} <- GenServer.start(__MODULE__, {space, dim, max_elements}) do
      {:ok,
       %T{
         space: space,
         dim: dim,
         pid: pid
       }}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Query the index with a single vector or a list of vectors.

  ##### Positional Parameters

  - *query*: `Nx.Tensor.t() | binary() | [binary()]`.

    A vector or a list of vectors to query.

    If *query* is a list of vectors, the vectors must be of the same dimension.

  ##### Keyword Paramters

  - *k*: `pos_integer()`.

    Number of nearest neighbors to return.
  """
  @spec knn_query(%T{}, Nx.Tensor.t() | binary() | [binary()], [
          {:k, pos_integer()}
          # {:filter, function()}
        ]) :: :ok | {:error, String.t()}
  def knn_query(self, query, opts \\ [])

  def knn_query(self = %T{}, query, opts) when is_binary(query) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         #  {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         :ok <- might_be_float_data?(query),
         features = trunc(byte_size(query) / HNSWLib.Nif.float_size()),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      GenServer.call(self.pid, {:knn_query, query, k, nil, 1, features})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query, opts) when is_list(query) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         {:ok, {rows, features}} <- Helper.list_of_binary(query),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      GenServer.call(
        self.pid,
        {:knn_query, IO.iodata_to_binary(query), k, filter, rows, features}
      )
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query = %Nx.Tensor{}, opts) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, query) do
      GenServer.call(self.pid, {:knn_query, f32_data, k, filter, rows, features})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Add items to the index.

  ##### Positional Parameters

  - *data*: `Nx.Tensor.t()`.

    Data to add to the index.

  ##### Keyword Parameters

  - *ids*: `Nx.Tensor.t() | [non_neg_integer()] | nil`.

    IDs to assign to the data.

    If `nil`, IDs will be assigned sequentially starting from 0.

    Defaults to `nil`.

  """
  @spec add_items(%T{}, Nx.Tensor.t(), [
          {:ids, Nx.Tensor.t() | [non_neg_integer()] | nil}
        ]) :: :ok | {:error, String.t()}
  def add_items(self, data, opts \\ [])

  def add_items(self = %T{}, data = %Nx.Tensor{}, opts) when is_list(opts) do
    with {:ok, ids} <- normalize_ids(opts[:ids]),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, data) do
      GenServer.call(self.pid, {:add_items, f32_data, ids, rows, features})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Delete vectors with the given labels from the index.
  """
  def delete_vector(self = %T{}, label) when is_integer(label) do
    GenServer.call(self.pid, {:delete_vector, label})
  end

  @doc """
  Save current index to disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to save the index to.
  """
  @spec save_index(%T{}, Path.t()) :: {:ok, integer()} | {:error, String.t()}
  def save_index(self = %T{}, path) when is_binary(path) do
    GenServer.call(self.pid, {:save_index, path})
  end

  @doc """
  Load index from disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to load the index from.

  ##### Keyword Parameters

  - *max_elements*: `non_neg_integer()`.

    Maximum number of elements to load from the index.

    If set to 0, all elements will be loaded.

    Defaults to 0.
  """
  @spec load_index(%T{}, Path.t(), [
          {:max_elements, non_neg_integer()}
        ]) :: :ok | {:error, String.t()}
  def load_index(self = %T{}, path, opts \\ []) when is_binary(path) and is_list(opts) do
    with {:ok, max_elements} <- Helper.get_keyword!(opts, :max_elements, :non_neg_integer, 0) do
      GenServer.call(self.pid, {:load_index, path, max_elements})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp verify_data_tensor(self = %T{}, data = %Nx.Tensor{}) do
    row_features =
      case data.shape do
        {rows, features} ->
          ensure_vector_dimension(self, features, {rows, features})

        {features} ->
          ensure_vector_dimension(self, features, {1, features})

        shape ->
          {:error,
           "Input vector data wrong shape. Number of dimensions #{tuple_size(shape)}. Data must be a 1D or 2D array."}
      end

    with {:ok, {rows, features}} <- row_features do
      {:ok, Nx.to_binary(Nx.as_type(data, :f32)), rows, features}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp ensure_vector_dimension(%T{dim: dim}, dim, ret), do: {:ok, ret}

  defp ensure_vector_dimension(self = %T{}, features, _ret) do
    {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
  end

  defp might_be_float_data?(data) do
    if rem(byte_size(data), HNSWLib.Nif.float_size()) != 0 do
      {:error,
       "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"}
    else
      :ok
    end
  end

  defp normalize_ids(ids = %Nx.Tensor{}) do
    case ids.shape do
      {_} ->
        {:ok, Nx.to_binary(Nx.as_type(ids, :u64))}

      shape ->
        {:error, "expect ids to be a 1D array, got `#{inspect(shape)}`"}
    end
  end

  defp normalize_ids(ids) when is_list(ids) do
    if Enum.all?(ids, fn x ->
         is_integer(x) and x >= 0
       end) do
      {:ok, ids}
    else
      {:error, "expect `ids` to be a list of non-negative integers"}
    end
  end

  defp normalize_ids(nil) do
    {:ok, nil}
  end

  # GenServer callbacks

  @impl true
  def init({space, dim, max_elements}) do
    case HNSWLib.Nif.bfindex_new(space, dim, max_elements) do
      {:ok, ref} ->
        {:ok, ref}

      {:error, reason} ->
        {:stop, {:error, reason}}
    end
  end

  @impl true
  def handle_call({:knn_query, data, k, filter, rows, features}, _from, self) do
    case HNSWLib.Nif.bfindex_knn_query(self, data, k, filter, rows, features) do
      {:ok, labels, dists, rows, k, label_bits, dist_bits} ->
        labels = Nx.reshape(Nx.from_binary(labels, :"u#{label_bits}"), {rows, k})
        dists = Nx.reshape(Nx.from_binary(dists, :"f#{dist_bits}"), {rows, k})
        {:reply, {:ok, labels, dists}, self}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def handle_call({:delete_vector, label}, _from, self) do
    {:reply, HNSWLib.Nif.bfindex_delete_vector(self, label), self}
  end

  @impl true
  def handle_call({:add_items, f32_data, ids, rows, features}, _from, self) do
    {:reply, HNSWLib.Nif.bfindex_add_items(self, f32_data, ids, rows, features), self}
  end

  @impl true
  def handle_call({:save_index, path}, _from, self) do
    {:reply, HNSWLib.Nif.bfindex_save_index(self, path), self}
  end

  @impl true
  def handle_call({:load_index, path, max_elements}, _from, self) do
    {:reply, HNSWLib.Nif.bfindex_load_index(self, path, max_elements), self}
  end
end
