defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :reference]
  alias __MODULE__, as: T

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
    with {:ok, m} <- get_keyword(opts, :m, :non_neg_integer, 16),
         {:ok, ef_construction} <- get_keyword(opts, :ef_construction, :non_neg_integer, 200),
         {:ok, random_seed} <- get_keyword(opts, :random_seed, :non_neg_integer, 100),
         {:ok, allow_replace_deleted} <-
           get_keyword(opts, :allow_replace_deleted, :boolean, false),
         {:ok, ref} <-
           HNSWLib.Nif.new(
             space,
             dim,
             max_elements,
             m,
             ef_construction,
             random_seed,
             allow_replace_deleted
           ) do
      {:ok,
       %T{
         space: space,
         dim: dim,
         reference: ref
       }}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp get_keyword(opts, key, type, default) do
    get_keyword(key, opts[key] || default, type)
  end

  defp get_keyword(_key, val, :non_neg_integer) when is_integer(val) and val >= 0 do
    {:ok, val}
  end

  defp get_keyword(key, val, :non_neg_integer) do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a non-negative integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :boolean) when is_boolean(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :boolean) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be a boolean, got `#{inspect(val)}`"}
  end
end
