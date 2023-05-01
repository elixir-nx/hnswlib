defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :pid]
  alias __MODULE__, as: T
  alias HNSWLib.Helper

  use GenServer

  @doc """
  Construct a new Index

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

  ##### Keyword Paramters

  - *m*: `non_neg_integer()`.
  - *ef_construction*: `non_neg_integer()`.
  - *random_seed*: `non_neg_integer()`.
  - *allow_replace_deleted*: `boolean()`.
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer(), non_neg_integer(), [
          {:m, non_neg_integer()},
          {:ef_construction, non_neg_integer()},
          {:random_seed, non_neg_integer()},
          {:allow_replace_deleted, boolean()}
        ]) :: {:ok, %T{}} | {:error, String.t()}
  def new(space, dim, max_elements, opts \\ [])
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_integer(max_elements) and max_elements >= 0 do
    with {:ok, m} <- Helper.get_keyword(opts, :m, :non_neg_integer, 16),
         {:ok, ef_construction} <-
           Helper.get_keyword(opts, :ef_construction, :non_neg_integer, 200),
         {:ok, random_seed} <- Helper.get_keyword(opts, :random_seed, :non_neg_integer, 100),
         {:ok, allow_replace_deleted} <-
           Helper.get_keyword(opts, :allow_replace_deleted, :boolean, false),
         {:ok, pid} <-
           GenServer.start(
             __MODULE__,
             {space, dim, max_elements, m, ef_construction, random_seed, allow_replace_deleted}
           ) do
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

  @spec knn_query(%T{}, Nx.Tensor.t() | binary() | [binary()], [
          {:k, pos_integer()},
          {:num_threads, integer()},
          {:filter, function()}
        ]) :: :ok | {:error, String.t()}
  def knn_query(self, data, opts \\ [])

  def knn_query(self = %T{}, data, opts) when is_binary(data) do
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true) do
      if rem(byte_size(data), float_size()) != 0 do
        {:error,
         "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"}
      else
        features = trunc(byte_size(data) / float_size())
        if features != self.dim do
          {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
        else
          GenServer.call(
            self.pid,
            {:knn_query, data, k, num_threads, filter, 1, features}
          )
        end
      end
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, data, opts) when is_list(data) do
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true),
         {:ok, {rows, features}} <- Helper.list_of_binary(data) do
      if features != self.dim do
        {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
      else
        GenServer.call(
          self.pid,
          {:knn_query, IO.iodata_to_binary(data), k, num_threads, filter, rows, features}
        )
      end
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, data = %Nx.Tensor{}, opts) do
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true) do
      case data.shape do
        {rows, features} ->
          if features != self.dim do
            {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
          else
            GenServer.call(
              self.pid,
              {:knn_query, Nx.to_binary(Nx.as_type(data, :f32)), k, num_threads, filter, rows,
               features}
            )
          end

        {features} ->
          if features != self.dim do
            {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
          else
            GenServer.call(
              self.pid,
              {:knn_query, Nx.to_binary(Nx.as_type(data, :f32)), k, num_threads, filter, 1,
               features}
            )
          end

        shape ->
          {:error,
           "Input vector data wrong shape. Number of dimensions #{tuple_size(shape)}. Data must be a 1D or 2D array."}
      end
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec get_ids_list(%T{}) :: {:ok, [integer()]} | {:error, String.t()}
  def get_ids_list(self = %T{}) do
    GenServer.call(self.pid, :get_ids_list)
  end

  defp float_size do
    HNSWLib.Nif.float_size()
  end

  # GenServer callbacks

  @impl true
  def init({space, dim, max_elements, m, ef_construction, random_seed, allow_replace_deleted}) do
    case HNSWLib.Nif.new(
           space,
           dim,
           max_elements,
           m,
           ef_construction,
           random_seed,
           allow_replace_deleted
         ) do
      {:ok, ref} ->
        {:ok, ref}

      {:error, reason} ->
        {:stop, {:error, reason}}
    end
  end

  @impl true
  def handle_call(:get_ids_list, _from, self) do
    {:reply, HNSWLib.Nif.get_ids_list(self), self}
  end

  @impl true
  def handle_call({:knn_query, data, k, num_threads, filter, rows, features}, _from, self) do
    case HNSWLib.Nif.knn_query(self, data, k, num_threads, filter, rows, features) do
      any ->
        {:reply, any, self}
    end
  end

  @impl true
  def handle_info({:knn_query_filter, filter, id}, _self) do
  end
end
